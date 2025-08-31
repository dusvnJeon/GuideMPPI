#pragma once

#include "guide_mppi_param.h"


#include <Eigen/Dense>
#include <EigenRand/EigenRand>   
#include <random>                
#include <chrono>          
#include <ctime>            
#include <cstdint>           
#include <fstream>         
#include <iostream>         
#include <vector>
#include <string>

#include "collision_checker.h"   
#include "matplotlibcpp.h"      

class GuideMPPI {
public:
    template<typename ModelClass>   // Model자체는 자료형이 다를 수 있음
    GuideMPPI(ModelClass model);
    ~GuideMPPI(); 

    void init(GuideMPPIParam mppi_param); 
    void setCollisionChecker(CollisionChecker *collision_checker);
    virtual Eigen::MatrixXd getNoise(const int &T);
    void move();
    Eigen::VectorXd solve();
    void show();
    void showTraj();

    Eigen::MatrixXd getTrajectory(std::string path, bool with_theta);
    int getNextWaypointIndex(const Eigen::VectorXd& cur_pos,const Eigen::MatrixXd& traj,int lastWaypointIndex);

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    std::chrono::duration<double> elapsed_1;        // 걸린 시간
    double elapsed;

    Eigen::MatrixXd U_0;        // 얘는 main.cpp에서 크기를 초기화해줌
    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    Eigen::MatrixXd Uo;         // u의 sequence 한 묶음
    Eigen::MatrixXd Xo;

    Eigen::MatrixXd X_ref;

    Eigen::MatrixXd extendRefTraj(const Eigen::MatrixXd& traj, int T, const Eigen::VectorXd& x_target);

    double acceptance_dist = 0.1;     // 기본값, init에서 덮어씀
    
    Eigen::VectorXd u0;

protected:
    int dim_x;      
    int dim_u;

    // Discrete Time System
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
    // Projection
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h; // U입력들을 프로젝션

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    // std::mt19937_64 urng{1};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    // Parameters
    float dt;
    int T;
    int N;  // # of time horizon
    double gamma_u; // temparature param
    Eigen::MatrixXd sigma_u;    // 노이즈 공분산
    
    CollisionChecker *collision_checker;



    std::vector<Eigen::VectorXd> visual_traj;

    int lastWaypointIndex;      // 이를 저장하며 글로벌 경로상에서 이 전에 있던 waypoint는 추종하지 않도록함.

    double getGuideCost(const Eigen::MatrixXd& Xi);  // 후보경로를 넣어주면 이와 X_ref 간의 GuideCost를 반환함.

    

    
};

template<typename ModelClass>
GuideMPPI::GuideMPPI(ModelClass model) {  // 걍 다 받아온 ModelClass꺼 그대로 쓰는거
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;

    this->f = model.f;
    this->q = model.q;
    this->p = model.p;
    this->h = model.h;
}

GuideMPPI::~GuideMPPI() {
}

void GuideMPPI::init(GuideMPPIParam mppi_param) {
    this->dt = mppi_param.dt;
    this->T = mppi_param.T;
    this->x_init = mppi_param.x_init;
    this->x_target = mppi_param.x_target;
    this->N = mppi_param.N;
    this->gamma_u = mppi_param.gamma_u;
    this->sigma_u = mppi_param.sigma_u;
    this->acceptance_dist = mppi_param.acceptance_dist;

    u0 = Eigen::VectorXd::Zero(dim_u);
    Xo = Eigen::MatrixXd::Zero(dim_x, T+1);
}

void GuideMPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd GuideMPPI::getNoise(const int &T) {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}   // 시간을 seed로 표준편차가 sigma_u안의 것을 쓴 랜덤변수 불러오기

Eigen::MatrixXd GuideMPPI::getTrajectory(std::string path, bool with_theta) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + path);
    }

    std::string line;
    std::vector<std::vector<double>> values;

    Eigen::MatrixXd ref_traj;

    bool first_line = true;
    while (std::getline(file, line)) {
        if (first_line) {  // 첫줄에 숫자 아닌 다른거 적혀있으면 그거 빼기
            first_line = false;
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {  // csv면 ','
            cell.erase(remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
            if (!cell.empty()) {
                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    std::cerr << "Warning: invalid cell -> " << cell << std::endl;
                }
            }
        }
        if (!row.empty()) values.push_back(row);
    }
    file.close();

    int rows = values.size();
    int cols = values[0].size();
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = values[i][j];

    if (!with_theta) {
        ref_traj = mat.leftCols(2);  // x,y
    } else {
        ref_traj = mat.leftCols(3);  // x,y,theta
    }
    return ref_traj.transpose();
}


void GuideMPPI::move() {
    x_init = x_init + (dt * f(x_init, u0));
    U_0.leftCols(T-1) = Uo.rightCols(T-1);
    U_0.col(T - 1).setZero();

}   // 진짜 x_init업데이트할때 씀

Eigen::VectorXd GuideMPPI::solve() {
    start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd Ui = U_0.replicate(N, 1);   // 외부 레퍼런스 입력
    Eigen::VectorXd costs(N);
    Eigen::VectorXd weights(N); 
    #pragma omp parallel for    // 해당 for에 대해서 CPU로 병렬로 실행
    for (int i = 0; i < N; ++i) {
        Eigen::MatrixXd Xi(dim_x, T+1);
        Eigen::MatrixXd noise = getNoise(T);    
        Ui.middleRows(i * dim_u, dim_u) += noise; 
        h(Ui.middleRows(i * dim_u, dim_u));     // 생성된 입력을 프로젝션함. 제한조건에 맞게

        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < T; ++j) {
            cost += p(Xi.col(j), x_target);
            // cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
            Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)));
        }

        cost += getGuideCost(Xi);        // X_ref는 timestep이 행이 커질수록 커지지만, Xi는 timestep이 열이 커질수록 커짐.

        cost += p(Xi.col(T), x_target);
        for (int j = 1; j < T+1; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
        }
        costs(i) = cost;        // input나오면 그걸로 롤아웃해서 state 계산하고 그걸로부터 해당 input의 cost까지 계산함
    }

    double min_cost = costs.minCoeff(); 
    weights = (-gamma_u * (costs.array() - min_cost)).exp();
    double total_weight =  weights.sum();
    weights /= total_weight;        

    Uo = Eigen::MatrixXd::Zero(dim_u, T);   
    for (int i = 0; i < N; ++i) {
        Uo += Ui.middleRows(i * dim_u, dim_u) * weights(i);
    }       // weight 계산 끝난 애들로 최종 U를 구함
    h(Uo);  // 이를 projection함. 사용가능한 범위로.


    finish = std::chrono::high_resolution_clock::now();
    elapsed_1 = finish - start;     // 걸린 시간

    elapsed = elapsed_1.count();    // count는 double로 변환해주는애

    u0 = Uo.col(0);     // 결국 계산된 U의 첫 행만 가져다 씀

    Xo.col(0) = x_init;
    for (int j = 0; j < T; ++j) {
        Xo.col(j+1) = Xo.col(j) + (dt * f(Xo.col(j), Uo.col(j)));
    }   // 현재까지 나온 최적 U sequence에 대한 rollout

    visual_traj.push_back(x_init);
    return u0;
}


void GuideMPPI::show() {     // 이번 solve로 내놓은 최적 궤적 하나에 대한 플롯
    namespace plt = matplotlibcpp;
    // plt::subplot(1,2,1);

    double resolution = 0.1;
    double hl = resolution / 2;
    for (int i = 0; i < collision_checker->map.size(); ++i) {
        for (int j = 0; j < collision_checker->map[0].size(); ++j) {
            if ((collision_checker->map[i])[j] == 10) {
                double mx = i*resolution;
                double my = j*resolution;
                std::vector<double> oX = {mx-hl, mx+hl, mx+hl, mx-hl, mx-hl};
                std::vector<double> oY = {my-hl,my-hl,my+hl,my+hl,my-hl};
                plt::plot(oX, oY, "k"); // oX, oY를 하나씩 매칭하면 플롯에 필요한 사각형의 꼭짓점과 초기점으로 5개의 점으로 구성됨
            }
        }
    }

    std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Xo.cols()));
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < Xo.cols(); ++j) {
            X_MPPI[i][j] = Xo(i, j);
        }
    }
    // std::string color = "C" + std::to_string(9 - index%10);
    plt::plot(X_MPPI[0], X_MPPI[1], {{"color", "black"}, {"linewidth", "10.0"}});


if (X_ref.size() > 0 && X_ref.rows() >= 2) {
    std::vector<double> RX(X_ref.cols()), RY(X_ref.cols());
    for (int j = 0; j < X_ref.cols(); ++j) {
        RX[j] = X_ref(0, j);  // x
        RY[j] = X_ref(1, j);  // y
    }
    plt::plot(RX, RY, {{"color","blue"},{"linestyle","--"},{"linewidth","1.5"}});
}

    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    plt::show();
}

void GuideMPPI::showTraj() {     // 실제로 지나온 footprint에 대한 궤적 플롯
    namespace plt = matplotlibcpp;

    double resolution = 0.1;
    double hl = resolution / 2;
    for (int i = 0; i < collision_checker->map.size(); ++i) {
        for (int j = 0; j < collision_checker->map[0].size(); ++j) {
            if ((collision_checker->map[i])[j] == 10) {
                double mx = i*resolution;
                double my = j*resolution;
                std::vector<double> oX = {mx-hl, mx+hl, mx+hl, mx-hl, mx-hl};
                std::vector<double> oY = {my-hl,my-hl,my+hl,my+hl,my-hl};
                plt::plot(oX, oY, "k");
            }
        }
    }

    std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(visual_traj.size()));
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < visual_traj.size(); ++j) {
            X_MPPI[i][j] = visual_traj[j](i);
        }
    }
    // std::string color = "C" + std::to_string(9 - index%10);
    // plt::plot(X_MPPI[0], X_MPPI[1], {{"color", "black"}, {"linewidth", "10.0"}});

    plt::plot(X_MPPI[0], X_MPPI[1], {{"color", "black"}, {"linewidth", "1.0"}});
    plt::plot(X_MPPI[0], X_MPPI[1], "ro");


    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    plt::show();
}

int GuideMPPI::getNextWaypointIndex(const Eigen::VectorXd& cur_pos,const Eigen::MatrixXd& traj,int lastWaypointIndex){

    Eigen::VectorXd all_dist(traj.cols());
    all_dist.setConstant(std::numeric_limits<double>::max());
    Eigen::Index minIndex;

    if (lastWaypointIndex >= traj.cols()) 
        return traj.cols() - 1;

    for(int i=lastWaypointIndex; i<traj.cols(); i++){
        all_dist(i) = (cur_pos - traj.col(i)).norm();
    }

    double min_val = all_dist.minCoeff(&minIndex);
    
    if (min_val < acceptance_dist){
        return minIndex+1;
    }
    else{
        return minIndex;
    }
}

Eigen::MatrixXd GuideMPPI::extendRefTraj(const Eigen::MatrixXd& traj, int T,const Eigen::VectorXd& x_target) {
    int rows = traj.rows();
    int cols = traj.cols();

    Eigen::MatrixXd extendedTraj(rows, cols + T);

    extendedTraj.leftCols(cols) = traj;

    for (int i = 0; i < T; i++) {
        extendedTraj.col(cols + i) = x_target.head(rows);
    }

    return extendedTraj;
}

double GuideMPPI::getGuideCost(const Eigen::MatrixXd& Xi){
    const int r = X_ref.rows();
    return (Xi.topRows(r) - X_ref).colwise().norm().sum();
}
