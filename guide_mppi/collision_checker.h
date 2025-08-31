#pragma once

/*
들어가기전)
예상되는 이 코드
0. 기능
    a. 결론적으로 구해진 path에 대한 충돌 체크
    b. 중간 중간 만든 mppi의 path에 대한 충돌 체크
1. input : path, map(or pos of obstacle)
2. output : True or False (if there were collision?)
3. function(input to output algorithm) : 
    a. path와 map을 grid로 변환하며, map의 grid map에는 장애물이 있는지 없는지에 따라 1이나 0이 채워진다.
    b. path의 grid map에는 에이전트가 해당 grid를 지나가는지 아닌지에 따라 1이나 0이 채워진다.
    c. 두 grid map의 각 grid를 비교하며 에이전트가 지나가는 grid에 장애물이 있을 경우 True를 반환하며, 그렇지 않을 경우 False를 반환한다.

+ npy_to_txt.py 코드에 의해 만들어진 map의 txt파일은 30행 50열이며, (1,2,29,30)행들은 1로, 벽을 의미하는 듯 하다.
    */


#include "model_base.h"

#include <eigen3/Eigen/Dense>

#include <vector>
#include <array>
#include <fstream>

class CollisionChecker {
public:
    CollisionChecker();
    ~CollisionChecker();

    void loadMap(const std::string& file_path, double resolution);
    void set3D(double max_hei); // ???

    // x, y, r, r^2
    std::vector<std::array<double, 4>> circles;
    // x1, x2, y1, y2
    std::vector<std::array<double, 4>> rectangles;

    void addCircle(double x, double y, double r);
    void addRectangle(double x, double y, double w, double h);

    bool getCollisionGrid(const Eigen::VectorXd &x);
    bool getCollisionCircle(const Eigen::VectorXd &z);
    bool getCollisionGrid_polygon(const Eigen::VectorXd &x);        // polygon : 맵기반이 아닌 사전에 등록한 기하 도형을 대상으로 하는 충돌체크
    bool getCollisionCircle_polygon(const Eigen::VectorXd &z);
    bool getCollisionGrid_map(const Eigen::VectorXd &x);
    bool getCollisionCircle_map(const Eigen::VectorXd &z);

    std::vector<std::vector<double>> map;
private:
    bool is_3d;         // 이건 뭐지?
    int max_hei;        // 이건 뭐지?
    bool with_map;      // loadMap이라는 함수를 사용했으면 이게 True로 바뀜.
    double resolution;
    int max_row;
    int max_col;
};

CollisionChecker::CollisionChecker() {
    circles.clear();
    rectangles.clear();
    with_map = false;
    is_3d = false;
}

CollisionChecker::~CollisionChecker() {
}

void CollisionChecker::loadMap(const std::string& file_path, double resolution) {
    map.clear();        // map 초기화
    std::ifstream file(file_path);      // input file stream. 파일을 읽기 전용으로 여는 클래스
    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> row;    // 각 요소가 double인 row라는 이름의 벡터 생성
        std::string::size_type sz = 0;

        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + file_path);
        }

        while (sz < line.length()) {    // line에 담겨있는 요소의 개수가 sz(=0)과 같아질 때까지
            double value = std::stod(line, &sz);    // line의 맨 앞에 있는 숫자를 value에 기록하고, 해당 숫자의 길이를 sz에 기록함
            row.push_back(value);       // value를 row에 추가. 향후에 map에 이를 추가함
            line = line.substr(sz);     // 읽어온 숫자의 길이만큼을 line에서 삭제함.
        }

        map.push_back(row); // row에 저장된 맵의 행 정보를 map에 추가함.
    }

    file.close();   // 다 읽어왔으면 그만 닫기

    with_map = true;    // 맵 읽어왔으니 이제 true지
    this->resolution = resolution;  // 해당 함수에서 resolution을 입력변수로 받는데, 이 자체를 객체 내부의 resolution 변수의 값으로 받음
    max_row = map.size();       // 행 개수
    max_col = map[0].size();    // 열 개수
}

void CollisionChecker::set3D(double max_hei) {
    this->is_3d = true;
    this->max_hei = max_hei/resolution;
}

void CollisionChecker::addCircle(double x, double y, double r) {
    circles.push_back({x, y, r, r*r});
}       // 맵에 장애물 임의로 추가

void CollisionChecker::addRectangle(double x, double y, double w, double h) {
    rectangles.push_back({x, x + w, y, y + h});
}       // 맵에 장애물 임의로 추가

bool CollisionChecker::getCollisionGrid(const Eigen::VectorXd &x) {
    if (with_map) {return getCollisionGrid_map(x);}
    else {return getCollisionGrid_polygon(x);}
}       // Grid에서 충돌 확인. 입력 인자에 맵상에서 특정 픽셀 위치가 있음

bool CollisionChecker::getCollisionCircle(const Eigen::VectorXd &x) {
    if (with_map) {return getCollisionCircle_map(x);}
    else {return getCollisionCircle_polygon(x);}
}       // 원 map에서 충돌확인. 입력 인자에 맵상에서 특정 픽셀 위치가 있음

bool CollisionChecker::getCollisionGrid_polygon(const Eigen::VectorXd &x) {
    double dx;
    double dy;
    double distance_2;
    // Circle
    for (int i = 0; i < circles.size(); ++i) {      // 모든 circle에 대해서 반복문 실행
        dx = circles[i][0] - x(0);      // 들어온 입력인자 픽셀에 대해 x차잇값 계산
        dy = circles[i][1] - x(1);      // 들어온 입력인자 픽셀에 대해 y차잇값 계산
        distance_2 = (dx * dx) + (dy * dy);     // 
        if (distance_2 <= circles[i][3]) {return true;} // 거리 제곱값을 비교해서 원의 반지름의 제곱이 거리의 제곱값보다 크면 충돌했으니 true반환
        else {continue;}
    }
    // Rectangle
    for (int i = 0; i < rectangles.size(); ++i) {
        if (x(0) < rectangles[i][0]) {continue;}    // continue : 이번 for문 넘어감 (이 네가지 중 하나라도 만족하면 충돌 판정)
        else if (rectangles[i][1] < x(0)) {continue;}  
        else if (x(1) < rectangles[i][2]) {continue;}  
        else if (rectangles[i][3] < x(1)) {continue;}  
        else {return true;}
    }       // 모든 사각형에 대해서 주어진 점이 사각형의 꼭짓점들 안에 겹친다면 장애물에 충돌한 것이니 true 반환
    return false;
}

bool CollisionChecker::getCollisionCircle_polygon(const Eigen::VectorXd &z) {
    Eigen::VectorXd zj;
    double dx;
    double dy;
    double distance_2;
    double dc;
    // Circle
    for (int i = 0; i < circles.size(); ++i) {
        dx = circles[i][0] - z(0);      // z에서 첫 요소는 로봇의 x값
        dy = circles[i][1] - z(1);      // z에서 두번째 요소는 로봇의 y값
        distance_2 = (dx * dx) + (dy * dy);
        dc = circles[i][2] + z(2);      // z에서 세 번째 요소는 로봇의 반지름 (원형 로봇 가정)
        if (distance_2 <= (dc*dc)) {return true;}       // 장애물의 반지름과 로봇의 반지름을 더한 값이 장애물의 중심으로부터 로봇의 중심까지의 거리보다 길다면 충돌
        else {continue;}
    }
    // Rectangle
    for (int i = 0; i < rectangles.size(); ++i) {
        if ((z(0) + z(2)) < rectangles[i][0]) {continue;}
        else if (rectangles[i][1] < (z(0) - z(2))) {continue;}
        else if ((z(1) + z(2)) < rectangles[i][2]) {continue;}
        else if (rectangles[i][3] < (z(1) - z(2))) {continue;}
        else {return true;}     // 네개의 점을 검사해서 원형 로봇의 edge이 안에 들
    }
    return false;
}

bool CollisionChecker::getCollisionGrid_map(const Eigen::VectorXd &x) {     // 맵에 대해서 장애물 충돌 확인
    // Need to check comparison double
    int nx = round(x(0)/resolution);
    int ny = round(x(1)/resolution);
    if (nx < 0 || max_row <= nx) {return true;}
    if (ny < 0 || max_col <= ny) {return false;}
    if (is_3d) {
        int nz = round(x(2)/resolution);
        if (nz < 0 || max_hei <= nz) {return false;}
    }
    if (map[nx][ny] == 10) {return true;}
    return false;
}       // 장애물이 있으면 map의 요소값이 10이고, 아니면 5인듯?

bool CollisionChecker::getCollisionCircle_map(const Eigen::VectorXd &z) {
    // Need to check comparison double
    int cx = round(z(0)/resolution);
    int cy = round(z(1)/resolution);
    double r = z(2)/resolution + 0.5;       // 원을 격자 셀단위로 바꾸때 주는 마진
    if (cx < 0 || max_row <= cx) {return true;}
    if (cy < 0 || max_col <= cy) {return true;}
    if (map[cx][cy] == 10) {return true;}
    if (map[cx][cy] == 5) {return false;}
    if (map[cx][cy] < r) {return true;}
    return false;
}