#pragma once

struct GuideMPPIParam{
    float dt;
    int T;
    int N;
    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    double acceptance_dist;     // 추종하던 글로벌 경로의 waypoint 도달판정 기준

};