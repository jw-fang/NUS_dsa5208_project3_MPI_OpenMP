//mpic++ -g -O0 -std=c++17 -fopenmp -pthread -Iinclude -Iinclude/eigen3 -ffp-contract=off lr.cpp -o logistic_newton
//nvc++ -O0 -std=c++17 -mp=gpu -gpu=nofma     -DEIGEN_DONT_VECTORIZE     -Iinclude -Iinclude/eigen3     $(mpic++ --showme:compile)     lr_GPU_longdouble.cpp     $(mpic++ --showme:link)     -o logistic_newton_GPU
#include <mpi.h>
#include <omp.h>
#include <cmath>   
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <Eigen/Dense>
#include "include/csv.h"


double hour, month, day, location_longitude, location_latitude, rained;
int mpi_rank, mpi_size, openmp_thread=omp_get_max_threads(), csv_len, t_add;
std::vector<double> csv_data, local_data;
std::vector<int> rank_len, rank_offset;
std::string output_folder ="./output_GPU";
std::string csv_path;
std::string header;
int converged = 0;
// std::vector<std::string> header_list; 

struct xy {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
};

std::string mode;
int iter=1000;
constexpr int col=6;
double l2=1;
double tol=1e-4;
Eigen::VectorXd Weight=Eigen::VectorXd::Zero(col);

// formula, clamping to prevent over and underflow

double sig(double z){
    if (z>40) {return 1;}
    if (z<-40) {return 0;} 
    return (std::exp(z) / (1.0 + (std::exp(z))));}

double log1p_z (double z) {
    if (z>40) {return z;}
    if (z<-40) {return 0;} 
    return std::log1p(std::exp(z));}


double gpu_loss(const double* features, const double* labels,
                const double* weights, int nrows, int ncols) {
    double loss = 0.0;
    #pragma omp target teams distribute parallel for map(to: features[0:nrows*ncols], labels[0:nrows], weights[0:ncols]) reduction(+:loss)
    for (int i = 0; i < nrows; i++) {
        double z = 0.0;
        for (int f = 0; f < ncols; f++) {
            z += weights[f] * features[i*ncols+f];}
        double softplus;
        if (z > 40.0) {softplus = z;}
        else if (z < -40.0) {softplus = 0.0;}
        else {softplus = log1p(exp(z));}
        loss += softplus - labels[i] * z;}
    return loss;}

void gpu_predict(const double* features, const double* weights,
                 double* predictions, int nrows, int ncols) {
    #pragma omp target teams distribute parallel for map(to: features[0:nrows*ncols], weights[0:ncols]) map(from: predictions[0:nrows])
    for (int i = 0; i < nrows; i++) {
        double z = 0.0;
        for (int f = 0; f < ncols; f++) {
            z += weights[f] * features[i*ncols+f];
        }
        double p;
        if (z > 40.0) {p = 1.0;}
        else if (z < -40.0) {p = 0.0;}
        else {p = 1.0 / (1.0 + exp(-z));}
        if (p > 0.5) {predictions[i] = 1.0;}
        else {predictions[i] = 0.0;}}
}

int main(int argc, char* argv[]) {

    //parse argumment
   for (int i=1; i<argc; i++){
        std::string arg=argv[i];
        if (arg.find("mode=")==0){mode=arg.substr(5);}
        if (arg.find("iter=")==0){iter=std::abs(std::stoi(arg.substr(5)));}
        if (arg.find("l2=")==0){l2=std::abs(std::stod(arg.substr(3)));}
        if (arg.find("tol=")==0){tol=std::abs(std::stod(arg.substr(4)));}
        if (arg.find("path=")==0){csv_path=arg.substr(5);}
        if (arg.find("output_folder=")==0){
            if (arg.substr(14)!=""){output_folder=arg.substr(14);}}
        }
    std::filesystem::create_directory(output_folder);
    if ((mode!="training")&&(mode!="fit")){
        std::cerr << "please set the mode to either training or fit. exiting...";
        return 1;}

    if (mode=="training"){
        if (csv_path=="") {csv_path=("./processed.csv");}}

    if (mode=="fit"){
	if (csv_path=="") {csv_path=("./test.csv");}}

    //initialisation of the MPI + OpenMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &t_add);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    std::ofstream output_txt(output_folder+"/metrics_mpi"+std::to_string(mpi_size)+"openmp_"+std::to_string(openmp_thread)+".txt", std::ios::app);
    
    if (t_add < MPI_THREAD_FUNNELED) {
    MPI_Abort(MPI_COMM_WORLD, 1);}
    // read csv, only rank 0
    if (mpi_rank==0) {
        output_txt << "iter,tol,loss\n";
        
        std::cout << "reading csv...\n";


        io::CSVReader<col> csv(csv_path);
        std::ifstream input_csv(csv_path);
        std::getline(input_csv, header);
        if (!header.empty() && header.back()=='\r'){header.pop_back();}
        // std::stringstream stream(header);
        // std::vector<std::string> column_name;
        // std::string value;
        // while (std::getline(stream, value, ',')) {
        //     column_name.push_back(value);
        // }

        csv.read_header(
            io::ignore_extra_column,
            "hour","month","day","location_longitude","location_latitude","rained");
        while (csv.read_row(hour, month, day, location_longitude, location_latitude, rained))
        {csv_data.insert(csv_data.end(), 
            {hour, month, day, location_longitude, location_latitude, rained});
        }
        
        // to calculate the offset and no. of rows to assign to each rank
        csv_len=csv_data.size()/col;
        int rows_per_rank=csv_len/mpi_size;
        int rows_remainder=csv_len%mpi_size;
        
        for (int i=0; i<mpi_size; i++){
            int expected=rows_per_rank;
            if (rows_remainder >0) {
                expected+=1;
                rows_remainder-=1;}
            rank_len.push_back(expected*col);}

        for (int i=0; i<mpi_size; i++){
            if (i==0){rank_offset.push_back(0);}
            else {rank_offset.push_back(rank_offset[i-1]+rank_len[i-1]);}
        }

    }
    
    rank_len.resize(mpi_size);
    rank_offset.resize(mpi_size);
    // distribute
    MPI_Bcast(rank_len.data(),mpi_size,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(rank_offset.data(),mpi_size,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&csv_len,1,MPI_INT,0,MPI_COMM_WORLD);
    
    local_data.resize(rank_len[mpi_rank]);
    
    MPI_Scatterv(
    csv_data.data(), //source
    rank_len.data(),
    rank_offset.data(),
    MPI_DOUBLE, //source type
    local_data.data(), //destination
    rank_len[mpi_rank],
    MPI_DOUBLE, //destination type
    0, //root
    MPI_COMM_WORLD);
    
    //convert to feature/label pair
    const int rank_rows=local_data.size()/6;
    xy xy_data;
    xy_data.x.resize(rank_rows);
    xy_data.y.resize(rank_rows);
    
    #pragma omp parallel for
    for (int i=0; i < rank_rows; i++){
        int o=i*col;
        xy_data.x[i]={1.0,
            local_data[o],
            local_data[o+1],
            local_data[o+2],
            local_data[o+3],
            local_data[o+4]};
        xy_data.y[i]=local_data[o+5];
    }

    // for gpu_offloading
    std::vector<double> gpu_x(rank_rows * col);
    for (int r = 0; r < rank_rows; ++r)
        for (int f = 0; f < col; ++f)
            gpu_x[r * col + f] = xy_data.x[r][f];
    const double* gpu_features = gpu_x.data();
    const double* gpu_labels = xy_data.y.data();


    std::vector<double> predict_label(rank_rows);
    //return predictions
    std::vector<int> row_counts(mpi_size),row_offsets(mpi_size);
    std::vector<double> all_predictions;

    for (int i = 0; i<mpi_size; i++) {
        row_counts[i] = rank_len[i]/col;
        row_offsets[i] = rank_offset[i] / col;}

    auto prediction = [&](const xy& xy_data) {
        const int nrows = rank_rows;
        double gpu_weights[col];
        for (int f = 0; f < col; ++f) gpu_weights[f] = Weight[f];
        gpu_predict(gpu_x.data(), gpu_weights, predict_label.data(), nrows, col);
        if (mpi_rank == 0) {all_predictions.resize(csv_len);}
        MPI_Gatherv(predict_label.data(),rank_rows,MPI_DOUBLE,
        all_predictions.data(),
        row_counts.data(),row_offsets.data(),
        MPI_DOUBLE,0, MPI_COMM_WORLD);
    };

    double recall, precision, f1;
    auto performance = [&](const std::vector<double>& y, const std::vector<double>& predict_label){
        double tp=0,fp=0,tn=0,fn=0;
        
        const double* actual = y.data();
        const double* predicted = predict_label.data();
        const int n = y.size();
        #pragma omp target teams distribute parallel for map(to: actual[0:n], predicted[0:n]) reduction(+:tp,fp,tn,fn)
        for (int i=0; i<n;i++){
            if (actual[i]==1.0 && predicted[i]==1.0){tp++;}
            if (actual[i]==0 && predicted[i]==1.0){fp++;}
            if (actual[i]==0 && predicted[i]==0){tn++;}
            if (actual[i]==1.0 && predicted[i]==0){fn++;}
        }

        std::vector<double> metrics={tp,fp,tn,fn};
        
        long double metrics_mpi[4];
        for (int j = 0; j < 4; ++j) {
            metrics_mpi[j] = metrics[j];
        }
        MPI_Allreduce(MPI_IN_PLACE, metrics_mpi, 4, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int j = 0; j < 4; ++j) {
            metrics[j] = metrics_mpi[j];
        }

        if (mpi_rank==0) {std::cout << std::fixed << std::setprecision(0) << "tp:"<<metrics[0]<<" fp:"<<metrics[1]<<" tn:"<<metrics[2]<<" fn:"<<metrics[3]<<'\n';}
        precision=metrics[0]/(metrics[0]+metrics[1]);
        recall=metrics[0]/(metrics[0]+metrics[3]);
        f1=2*((recall*precision)/(recall+precision));

    };
    // training start
    if (mode=="training"){
		if(mpi_rank==0){
		std::cout << "training...\n" ;}
    // fitting start
    Eigen::VectorXd Gradient=Eigen::VectorXd::Zero(col);
    Eigen::MatrixXd Hessian=Eigen::MatrixXd::Zero(col,col);
    Eigen::VectorXd Step=Eigen::VectorXd::Zero(col);


    for (int i=0; i<=iter; i++){
        double grad_flat[col] = {0};
        double hess_flat[col * col] = {0};

        Gradient=Gradient.setZero();
        Hessian=Hessian.setZero();
        Step=Step.setZero();
        //calculate grad and hessian

        const int nrows = rank_rows;
        double gpu_weights[col];
        for (int f = 0; f < col; ++f) gpu_weights[f] = Weight[f];
        #pragma omp target teams distribute parallel for map(to: gpu_features[0:rank_rows*col], gpu_labels[0:rank_rows]) map(to: gpu_weights[0:col]) reduction(+:grad_flat[0:col],hess_flat[0:col*col])
        for (int row=0; row<nrows; row++){
            double z = 0.0;
            for (int f=0; f<col; f++) z += gpu_weights[f] * gpu_features[row*col+f];
            double p;
            if (z > 40.0) {p = 1.0;}
            else if (z < -40.0) {p = 0.0;}
            else {p = 1.0 / (1.0 + exp(-z));}
            const double err = p - gpu_labels[row];
            const double curv = p * (1.0 - p);
            for (int f=0; f<col; f++) {
                const double xf = gpu_features[row*col+f];
                grad_flat[f] += err * xf;
                for (int f2=f; f2<col; f2++) {
                    const double value = curv * xf * gpu_features[row*col+f2];
                    hess_flat[f*col+f2] += value;
                    if (f != f2) hess_flat[f2*col+f] += value;
                }
            }
        }
        //collapse all ranks into 1 result, must wait for other ranks
        long double grad_mpi[col];
        long double hess_mpi[col * col];
        for (int f = 0; f < col; ++f) {
            grad_mpi[f] = grad_flat[f];
        }
        for (int f = 0; f < col * col; ++f) {
            hess_mpi[f] = hess_flat[f];
        }
        MPI_Allreduce(MPI_IN_PLACE, grad_mpi, col, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, hess_mpi, col*col, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        

        for (int f=0; f<col; f++) {
            Gradient[f] = grad_mpi[f] / csv_len;
            if (f!=0){
                Gradient[f] += (Weight[f] / (l2 * csv_len));
                Hessian(f, f) += 1 / (l2 * csv_len);}

            for (int f2 = 0; f2<col; f2++){
                Hessian(f, f2) += hess_mpi[f * col + f2] / csv_len;
        }}

        //loss 
        auto loss_fx = [&](const auto& Weight) {
            const int nrows = rank_rows;
            double loss = 0.0;
            double l2_loss = 0.0;
            double trial_weights[col];
            for (int f=0; f<col; ++f) trial_weights[f] = Weight[f];
            loss = gpu_loss(gpu_features, gpu_labels, trial_weights, nrows, col);

            long double loss_mpi = loss;
            MPI_Allreduce(MPI_IN_PLACE, &loss_mpi, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            for (int f = 1; f < col; ++f) {
                l2_loss += Weight[f] * Weight[f];
            }

            return (loss_mpi / csv_len) + (l2_loss / (2.0 * l2 * csv_len));
        };
        
        // evaluation
        
        double decrement = 0;
        double final_loss = 0;
        double max_g = 0;
        
        final_loss = loss_fx(Weight);
 
        if (mpi_rank == 0) {
        Eigen::LLT<Eigen::MatrixXd> cholesky(Hessian);
        if (cholesky.info() != Eigen::Success) {
            if (mpi_rank == 0){
            std::cerr << "Hessian not PD, terminating...\n";}
            MPI_Abort(MPI_COMM_WORLD, 1);}
        
        Step = cholesky.solve(Gradient);
        max_g = Gradient.cwiseAbs().maxCoeff();
        decrement = Gradient.dot(Step);
        
        std::cout << "iter " << i <<": tol:" << max_g << " loss:" << final_loss << '\n';

        output_txt << i << "," << max_g << "," << final_loss << '\n';
        if ((max_g <= tol) && (0.5*decrement <= tol)){
            converged=1;
            std::cout << "converged at iter " << i << '\n';}
        }
        
        MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (converged==1){
            prediction(xy_data);
            performance(xy_data.y,predict_label);
            if (mpi_rank == 0) {std::cout<< "performance: precision:"<< precision
            <<" recall:"<<recall<<" f1:"<<f1<<'\n';}
            break;}

        
        MPI_Bcast(&decrement, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&final_loss, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(Step.data(),col,MPI_DOUBLE,0,MPI_COMM_WORLD);

        //proceeding to next iter
        double t = 1.0;
        while (loss_fx(Weight-(t*Step)) > (final_loss-(1.0/2048.0)*t* decrement)) {
            t /= 2;
            if (t < 1e-12) {
                if (mpi_rank == 0){std::cerr << "line search failed\n";}
                MPI_Abort(MPI_COMM_WORLD, 1);}
            }
        

        Weight -= t * Step;
    }

    if ((mpi_rank == 0) && (converged==0)) { 
        std::cout << "did not converge at iter " << iter << " , terminating...\n";}
    }
    //training end

    //fit start
    if (mode=="fit"){
        //load existing weights
        std::ifstream weight_file(output_folder+"/weights.txt");
        std::string current_weight;
        for (int w=0; w < col;  w++) {
			std::getline(weight_file, current_weight, ',');
			Weight[w]=std::stod(current_weight);
			}
        prediction(xy_data);
        performance(xy_data.y,predict_label);
        if (mpi_rank == 0) {
            std::ofstream prediction_txt(output_folder+"/prediction.txt", std::ios::trunc);
            std::ifstream input_csv(csv_path);
            prediction_txt << header <<",prediction"<<'\n';
            std::string line;
            std::getline(input_csv, line);
            for (int i=0; i < all_predictions.size(); i++) {
            std::getline(input_csv, line);
			if (!line.empty() && line.back()=='\r'){line.pop_back();}
            prediction_txt << line << "," << all_predictions[i] << "\n";
        }
            
            std::cout<< "performance: precision:"<< precision
            <<" recall:"<<recall<<" f1:"<<f1<<'\n';
        }

    }
    //fit end
    

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double max;
    double running = end-start;

    MPI_Reduce(&running,&max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::string time="time_mpi"+std::to_string(mpi_size)+"openmp_"+std::to_string(openmp_thread)+".txt";
        std::ofstream time_txt(output_folder+"/"+time, std::ios::app);
        std::cout <<"longest duration: " << max << "s"<<'\n';
        time_txt << max <<'\n';

        //write weights
        if (mode=="training" && converged==1){
            std::ofstream weight_file(output_folder+"/weights.txt", std::ios::trunc);
            std::ofstream weight_mpi(output_folder+"/weights_mpi"+std::to_string(mpi_size)+"openmp_"+std::to_string(openmp_thread)+".txt", std::ios::app);
            weight_file << std::fixed << std::setprecision(8);
            weight_mpi << std::fixed << std::setprecision(8);
            for (int w=0; w < Weight.size(); w++) {
            weight_file << Weight[w] << ',';
            weight_mpi << Weight[w] << ',';}
            weight_mpi << '\n';

        }
    }


    MPI_Finalize();
    return 0;
}