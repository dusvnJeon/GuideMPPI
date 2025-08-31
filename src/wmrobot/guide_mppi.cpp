#include <wmrobot_map.h>
#include "GuideMPPI.h"
#include "collision_checker.h"

#include <iostream>
#include <Eigen/Dense>



int main(){
    auto model = WMRobotMap();

    using Solver = GuideMPPI;        
    using SolverParam = GuideMPPIParam;

    SolverParam param;     
    param.dt = 0.1;    
    param.T = 100;      // time horizon
    param.x_init.resize(model.dim_x);
    param.x_init << 0.5, 0.0, M_PI_2;
    param.x_target.resize(model.dim_x);
    param.x_target << 0.5, 5.0, M_PI_2;
    param.N = 6000;     // 샘플은 6천개
    param.gamma_u = 10.0;       // 온도파라미터
    Eigen::VectorXd sigma_u(model.dim_u);       // 노이즈 정도
    sigma_u << 0.25, 0.25;      // 0.25로 설정
    param.sigma_u = sigma_u.asDiagonal();       // 대각행렬로 만듦
    param.acceptance_dist = 0.1;
    double f_err;

    int maxiter = 200;      // 최대 시도는 200번

    bool is_collision = false;

    double total_elapsed = 0.0;
    
    Eigen::VectorXd u;


    CollisionChecker collision_checker = CollisionChecker(); 
    collision_checker.loadMap("../BARN_dataset/txt_files/output_300.txt", 0.1);     // 0.1은 해상도임
    Solver solver(model); 
    solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);


    std::string path = "../ref_traj/reference_trajectory_78.csv";   // Guide reference trajectory path
    bool with_theta = true;

    Eigen::MatrixXd traj = solver.getTrajectory(path, with_theta);

		traj = solver.extendRefTraj(traj,param.T+1,param.x_target);
		
		int lastwaypoint = 0;
		int nextwaypoint = 0;

        bool is_success = false;
		
		
		for(int i=0; i<maxiter; i++){
	
			nextwaypoint = solver.getNextWaypointIndex(solver.x_init,traj,lastwaypoint);

			solver.X_ref = traj.middleCols(nextwaypoint, param.T+1);
            // std::cout << nextwaypoint << std::endl;
            std::cout << solver.X_ref.col(0).transpose() << std::endl;
		
                u = solver.solve();
                solver.move();      // move하면 x_init이 한칸 앞으로 가게 됨

                total_elapsed += solver.elapsed;        // 이동에 걸린 시간 축적

                if (collision_checker.getCollisionGrid(solver.x_init)) {
                    is_collision = true;
                    break;
                }       // move된 점에 대해 collision 비교. 충돌했으면 그만 iter하기
                else {
                    f_err = (solver.x_init - param.x_target).norm();
                    // std::cout<<"f_err = "<<f_err<<std::endl;
                    if (f_err < 0.1) {
                        is_success = true;

                        std::cout << f_err << std::endl;
                        break;
                    }       // 만약 maxiter안에 한계 내에 도달했다면 성공! 
                }
                lastwaypoint = nextwaypoint;
                std::cout << i << std::endl;
                std::cout<< u <<std::endl;
                
                // solver.show();
            }

            solver.showTraj();
            std::cout<<'\t'<<is_success<<'\t'<<'\t'<<total_elapsed<<std::endl;
            // std::cout << solver.x_init.transpose() << std::endl;

            // std::cout<<traj<<std::endl;

            return 0;
		
		}
