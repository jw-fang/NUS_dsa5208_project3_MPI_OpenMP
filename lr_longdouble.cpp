//mpic++ -g -O0 -std=c++17 -fopenmp -pthread -Iinclude -Iinclude/eigen3 -ffp-contract=off lr.cpp -o logistic_newton
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


long double hour, month, day, location_longitude, location_latitude, rained;
int mpi_rank, mpi_size, openmp_thread=omp_get_max_threads(), csv_len, t_add;
std::vector<long double> csv_data, local_data;
std::vector<int> rank_len, rank_offset;
std::string output_folder ="./output";
std::string csv_path;
std::string header;
int converged = 0;
// std::vector<std::string> header_list; 

struct xy {
    std::vector<std::vector<long double>> x;
    std::vector<long double> y;
};

std::string mode;
int iter=1000;
constexpr int col=6;
long double l2=1;
long double tol=1e-4;
Eigen::VectorX<double long> Weight=Eigen::VectorX<double long>::Zero(col);

// formula, clamping to prevent over and underflow

long double sig(long double z){
    if (z>40) {return 1;}
    if (z<-40) {return 0;} 
    return (std::exp(z) / (1.0 + (std::exp(z))));}

long double log1p_z (long double z) {
    if (z>40) {return z;}
    if (z<-40) {return 0;} 
    return std::log1p(std::exp(z));}

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
        return 1;
    }

    if (mode=="training"){
        if (csv_path=="") {csv_path=("./processed.csv");}
    }

    if (mode=="fit"){
	if (csv_path=="") {csv_path=("./test.csv");}}

    //initialisation of the MPI + OpenMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &t_add);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Barrier(MPI_COMM_WORLD);
    long double start = MPI_Wtime();
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
        if (header.back()=='\r'){header.pop_back();}
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
    MPI_LONG_DOUBLE, //source type
    local_data.data(), //destination
    rank_len[mpi_rank],
    MPI_LONG_DOUBLE, //destination type
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

    std::vector<long double> predict_label(rank_rows);
    //return predictions
    std::vector<int> row_counts(mpi_size),row_offsets(mpi_size);
    std::vector<long double> all_predictions;

    for (int i = 0; i<mpi_size; i++) {
        row_counts[i] = rank_len[i]/col;
        row_offsets[i] = rank_offset[i] / col;}

    auto prediction = [&](const xy& xy_data) {
        #pragma omp parallel for
        for (int i = 0; i < rank_rows; i++) {
            long double z = 0;
            long double p;
            for (int w = 0; w < Weight.size(); w++) {
                z += Weight[w] * xy_data.x[i][w];}
            p = sig(z);
            if (p > 0.5){predict_label[i]=1;}
            else {predict_label[i]=0;}
        }
        if (mpi_rank == 0) {all_predictions.resize(csv_len);}

        MPI_Gatherv(predict_label.data(),rank_rows,MPI_LONG_DOUBLE,
        all_predictions.data(),
        row_counts.data(),row_offsets.data(),
        MPI_LONG_DOUBLE,0, MPI_COMM_WORLD);
    };
    
    
    long double recall, precision, f1;
    auto performance = [&](const std::vector<long double>& y, const std::vector<long double>& predict_label){
        long double tp=0,fp=0,tn=0,fn=0;
        
        #pragma omp parallel for reduction(+:tp,fp,tn,fn)
        for (int i=0; i<y.size();i++){
            if (y[i]==1.0 && predict_label[i]==1.0){tp++;}
            if (y[i]==0 && predict_label[i]==1.0){fp++;}
            if (y[i]==0 && predict_label[i]==0){tn++;}
            if (y[i]==1.0 && predict_label[i]==0){fn++;}
        }

        std::vector<long double> metrics={tp,fp,tn,fn};
        
        MPI_Allreduce(MPI_IN_PLACE, metrics.data(), 4, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
    Eigen::VectorX<double long> Gradient=Eigen::VectorX<double long>::Zero(col);
    Eigen::MatrixX<double long> Hessian=Eigen::MatrixX<double long>::Zero(col,col);
    Eigen::VectorX<double long> Step=Eigen::VectorX<double long>::Zero(col);


    for (int i=0; i<=iter; i++){
        //set to 0 at every loop
        long double grad[col] = {0};
        long double hess[col][col] = {{0}};

        Gradient=Gradient.setZero();
        Hessian=Hessian.setZero();
        Step=Step.setZero();
        //calculate grad and hessian

        #pragma omp parallel for reduction(+ : grad, hess)
        for (int row=0; row<rank_rows; row++){
            const auto& x = xy_data.x[row]; // loop thru rows
            long double z = 0; //initial
            for (int f=0; f<col; f++) {
                z += Weight[f] * x[f];}
            const long double p = sig(z); //prob
            const long double err = p - xy_data.y[row];
            const long double curv = p * (1 - p);
            for (int f=0; f<col; f++) {
                grad[f] += err * x[f];
                for (int f2 = f; f2 < col; f2++) {
                    long double value = curv * x[f] * x[f2];
                    hess[f][f2] += value;
                    if (f != f2) {
                        hess[f2][f] += value;}
                    }
                }
            }
        //collapse all ranks into 1 result, must wait for other ranks
        MPI_Allreduce(MPI_IN_PLACE, grad, col, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, hess, col*col, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        #pragma omp parallel for
        for (int f=0; f<col; f++) {
            Gradient[f] = grad[f]/csv_len;
            if (f!=0){
                Gradient[f] += (Weight[f] / (l2 * csv_len));
                Hessian(f, f) += 1 / (l2 * csv_len);}

            for (int f2 = 0; f2<col; f2++){
                Hessian(f, f2) += (hess[f][f2] / csv_len);
        }}

        //loss 
        auto loss_fx = [&](const auto& Weight) {
            long double loss = 0.0;
            long double l2_loss = 0.0;
            #pragma omp parallel for reduction(+:loss)
            for (int i = 0; i < rank_rows; i++) {
                long double z = 0.0;
                for (int f = 0; f < col; ++f) {
                    z += Weight[f] * xy_data.x[i][f];
                }
                loss += log1p_z(z) - xy_data.y[i] * z;
            }

            MPI_Allreduce(MPI_IN_PLACE, &loss, 1,MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            for (int f = 1; f < col; ++f) {
                l2_loss += Weight[f] * Weight[f];
            }

            return (loss / csv_len) + (l2_loss / (2.0 * l2 * csv_len));
        };
        
        // evaluation
        
        long double decrement = 0;
        long double final_loss = 0;
        long double max_g = 0;
        
        final_loss = loss_fx(Weight);
 
        if (mpi_rank == 0) {
        Eigen::LLT<Eigen::MatrixX<double long>> cholesky(Hessian);
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

        
        MPI_Bcast(&decrement, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&final_loss, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(Step.data(),col,MPI_LONG_DOUBLE,0,MPI_COMM_WORLD);

        //proceeding to next iter
        long double t = 1.0;
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
			if (line.back()=='\r'){line.pop_back();}
            prediction_txt << line << "," << all_predictions[i] << "\n";
        }
            
            std::cout<< "performance: precision:"<< precision
            <<" recall:"<<recall<<" f1:"<<f1<<'\n';
        }

    }
    //fit end
    
    MPI_Barrier(MPI_COMM_WORLD);
    long double end = MPI_Wtime();
    long double max;
    long double running = end-start;

    MPI_Reduce(&running,&max,1,MPI_LONG_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
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