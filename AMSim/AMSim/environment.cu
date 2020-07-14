#include "environment.cuh"
#include <iostream>
#include "display.cuh"
#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <ctime>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_types.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace Eigen;

extern const float SCALE_FACTOR = 40;
float scaledWindowWidth = 12;
float scaledWindowHeight = 12;
float windowWidth = scaledWindowWidth * SCALE_FACTOR;
float windowHeight = scaledWindowHeight * SCALE_FACTOR;

float t = 0; // current time
float dt = 0.01f; // time step
int n = 1; // total amt of learner matter
int n_t = 32; // total amt of teacher matter
int totalCount = n + n_t;

float pl = 60;
float rotDif = V / pl;
float transDif = pow(RADIUS, 2) * rotDif / 3;
float rotDifCoef = sqrt(2 * rotDif);
float transDifCoef = sqrt(2 * transDif);
float alpha = transDif / rotDif * pow(RADIUS, 2);
float mu = alpha * RADIUS / pl;
float range = 3;
float c = 1;
float s = 3;

float env_pi = 3.14159265358979323846;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0, 1);

__device__ float InteractionForce(float);
float Distance(float, float);
float DistancePBC(Matter, Matter, float&, float&);
float RadiansDifference(float, float);
__global__ void global_Movement(Matter*, Matter*, float*, int, float);
void DrawMatter(Matter, float = 0, float = 0);
Vector2f RandomOrt();
Vector2f RandomPos();

__device__ float d_mu;
__device__ float d_rotDifCoef;
__device__ float d_transDifCoef;
__device__ float d_sWidth;
__device__ float d_sHeight;
__device__ float d_range;

Environment::Environment(float Lnum, float Tnum, bool transientEnabled)
{
	cudaMemcpyToSymbol(&d_mu, &mu, sizeof(float*), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&d_rotDifCoef, &rotDifCoef, sizeof(float*), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&d_transDifCoef, &transDifCoef, sizeof(float*), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&d_sWidth, &scaledWindowWidth, sizeof(float*), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&d_sHeight, &scaledWindowHeight, sizeof(float*), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&d_range, &range, sizeof(float*), size_t(0), cudaMemcpyHostToDevice);

	srand(time(NULL));
	n = Lnum;
	n_t = Tnum;
	totalCount = n + n_t;
	for (int i = 0; i < sqrt(totalCount); i++) {
		for (int j = 0; j < sqrt(totalCount); j++) {
			if (Tnum + Lnum == 0)
				break;
			Vector2f ort = RandomOrt();
			float x = i - (windowWidth / 2 / SCALE_FACTOR / 2);
			float y = j - (windowHeight / 2 / SCALE_FACTOR / 2);
			if (Tnum > 0) {
				AddMatter(teacher, x, y, ort(0), ort(1));
				Tnum--;
			} else {
				AddMatter(learner, x, y, ort(0), ort(1));
				Lnum--;
			}
		}
	}

	// transient phase
	/*if (transientEnabled) {
		while (t < 10) {
			for (int i = 0; i < matters.size(); i++) {
				Matter &p = matters[i];
				Movement(p, 0);
			}
			t += dt;
			prevMatters = matters;
		}
		for (int i = 0; i < matters.size(); i++) {
			Matter &p = matters[i];
			p.acmlActiveWork = 0;
		}
		t = 0;
	}*/
}

std::vector<float> Environment::ReturnState()
{
	std::vector<float> state;
	for (int i = n_t; i < matters.size(); i++) {
		Matter &p = matters[i];

		int inRangeCount = 0;
		float totalRad = 0;
		float ortx = 0;
		float orty = 0;
		for (int j = 0; j < prevMatters.size(); j++)
		{
			Matter &m = prevMatters[j];
			float dx = 0;
			float dy = 0;
			float dis = DistancePBC(p, m, dx, dy);

			if (dis == 0)
				continue;

			if (dis < range) {
				ortx += m.ort[0];
				orty += m.ort[1];
				inRangeCount++;
			}
		}
		float radDiff = 0;
		if (inRangeCount != 0)
			radDiff = RadiansDifference(atan2(orty, ortx), atan2(p.ort[1], p.ort[0]));

		state.push_back(radDiff);
	}
	return state;
}

std::vector<float> Environment::Step(std::vector<float> actionList, std::vector<float> &rewardList, bool &terminate)
{
	Movement(actionList);

	t += dt;
	prevMatters = matters;

	//for (int i = n_t; i < matters.size(); i++) {
	//	Matter &p = matters[i];

	//	#pragma region Compute reward by neighbour lost
	//	int inRangeCount = 0;
	//	for (int j = 0; j < prevMatters.size(); j++) {
	//		Matter &m = prevMatters[j];
	//		float dx = 0;
	//		float dy = 0;
	//		float dis = DistancePBC(p, m, dx, dy);

	//		if (dis == 0)
	//			continue;

	//		if (dis < range)
	//			inRangeCount++;
	//	}

	//	if (inRangeCount < p.neighbourCount)
	//		rewardList[i-n_t] = (inRangeCount - p.neighbourCount)*c;
	//	else
	//		rewardList[i-n_t] = 0;
	//	p.neighbourCount = inRangeCount;
	//	#pragma endregion

	//}

	#pragma region Compute active work
	float normActiveWork =  returnCurrentActiveWork();
	float scaledActiveWork = powf((normActiveWork / dt)*10, 2);
	if (normActiveWork > 2 || normActiveWork < -2) terminate = true;
	//float G = 1 / dt * std::log(std::exp(-s*dt*totalCount*scaledActiveWork));
	//float activeWork_s = -(G / totalCount);

	std::fill(rewardList.begin(), rewardList.end(), scaledActiveWork);
	//for (int i = n_t; i < matters.size(); i++) {
	//	Matter &p = matters[i];
	//	rewardList[i - n_t] += powf((p.acmlCurrentActiveWork / dt) * 10, 2);;
	//}
	#pragma endregion

	return ReturnState();
}


void Environment::Movement(std::vector<float> actionList) {

	int count = matters.size();
	float* h_action;
	h_action = (float*)malloc(count * sizeof(float));
	for (int i = 0; i < matters.size(); i++) {
		h_action[i] = matters[i].type == learner ? actionList[i - n_t] : 0;
	}

	Matter* d_matters;
	Matter* h_matters;
	h_matters = (Matter*)malloc(count * sizeof(Matter));
	h_matters = &matters[0];
	Matter* d_prevMatters;
	Matter* h_prevMatters;
	h_prevMatters = (Matter*)malloc(count * sizeof(Matter));
	h_prevMatters = &prevMatters[0];
	float* d_action;

	std::cout << count * sizeof(Matter) << std::endl;

	if(cudaMalloc((void**)&d_action, count * sizeof(float)) != cudaSuccess) std::cout <<  cudaGetErrorString(cudaGetLastError()) << std::endl;
	if(cudaMalloc((void **)&d_matters, count * sizeof(Matter)) != cudaSuccess) std::cout << "failed 1!" << std::endl;
	if(cudaMalloc((void**)&d_prevMatters, count * sizeof(Matter)) != cudaSuccess) std::cout << "failed 2!" << std::endl;

	std::cout << count * sizeof(Matter) << std::endl;

	if(cudaMemcpy(d_matters, h_matters, count * sizeof(Matter), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "mem1 copy failed!" << std::endl;
	if(cudaMemcpy(d_prevMatters, h_prevMatters, count * sizeof(Matter), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "mem2 copy failed!" << std::endl;
	if(cudaMemcpy(d_action, h_action, count * sizeof(Matter), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "mem3 copy failed!" << std::endl;

	global_Movement<<<(count + 255) / 256, 256 >>>(d_matters, d_prevMatters, d_action, count, 0.01);

	if(cudaMemcpy(h_matters, d_matters, count * sizeof(Matter), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "mem copy out failed!" << std::endl;
	matters.assign(h_matters, h_matters + count);


	cudaFree(d_matters);
	cudaFree(d_prevMatters);
	cudaFree(d_action);
	//free(h_matters);
}

__global__ void global_Movement(Matter* d_matters, Matter* d_prevMatters, float* d_action, int count, float d_dt) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count) {
		Matter p = d_matters[i];
		float action = d_action[i];

		curandState state;
		curand_init(0, i, 0, &state);
		float eta = curand_normal(&state);
		float xi_1 = curand_normal(&state);
		float xi_2 = curand_normal(&state);

		float avgOrt = 0;
		float ortx = 0;
		float orty = 0;

		float r_0 = p.pos[0];
		float r_1 = p.pos[1];
		float F_0 = 0;
		float F_1 = 1;

		for (int j = 0; j < count; j++) {
			Matter m = d_prevMatters[j];

			float dx = p.pos[0] - m.pos[0];
			float dy = p.pos[1] - m.pos[1];

			#pragma region PBC Logic
			if (dx > d_sWidth / 2)
				dx -= d_sWidth;
			else if (dx <= -d_sWidth / 2)
				dx += d_sWidth;

			if (dy > d_sHeight / 2)
				dy -= d_sHeight;
			else if (dy <= -d_sHeight / 2)
				dy += d_sHeight;
			#pragma endregion

			float dis = sqrt(pow(dx, 2) + pow(dy, 2));

			if (dis == 0)
				continue;

			if (dis < d_range && m.type != learner) {
				ortx += m.ort[0];
				orty += m.ort[1];
			}

			float Fr = InteractionForce(dis);
			float forceAngle = atan2(dy, dx);
			F_0 += Fr * cos(forceAngle);
			F_1 += Fr * sin(forceAngle);
		}
		float rad = atan2(p.ort[1], p.ort[0]); // convert ort vector to radians
		float theta = 0;
		if (p.type == teacher) {
			float radDiff = 0;

			float radA = atan2(orty, ortx);
			float radB = rad;

			float pi = 3.14159265358979323846;

			radA += pi;
			radB += pi;
			float d = fmodf(abs(radA - radB), (float)pi * 2);
			float r = d > pi ? pi * 2 - d : d;
			if ((radA - radB >= 0 && radA - radB <= pi) || (radA - radB <= -pi && radA - radB >= -pi * 2))
				radDiff = r;
			radDiff = -r;

			if (ortx == 0 && orty == 0)
				radDiff = 0;
			theta = rad + radDiff * d_dt + sqrt(d_dt) * (d_rotDifCoef * eta);
		}
		else {
			theta = rad + action * d_dt + sqrt(d_dt) * (d_rotDifCoef * eta);
		}

		float u_0 = cos(theta);
		float u_1 = sin(theta);

		r_0 = r_0 + d_dt * (d_mu * F_0) + d_dt * (p.v * u_0) + sqrt(d_dt) * (d_transDifCoef * xi_1);
		r_1 = r_1 + d_dt * (d_mu * F_1) + d_dt * (p.v * u_1) + sqrt(d_dt) * (d_transDifCoef * xi_2);

		#pragma region Compute active work
		float r_aw_0 = p.pos[0];
		float r_aw_1 = p.pos[1];
		r_aw_0 = (d_mu * F_0) + (p.v * u_0) + (d_transDifCoef * xi_1);
		r_aw_1 = (d_mu * F_1) + (p.v * u_1) + (d_transDifCoef * xi_2);
		float aw = d_dt * (r_aw_0 * u_0 + r_aw_1 * u_1);
		p.acmlActiveWork += aw;
		p.acmlCurrentActiveWork = aw;
		#pragma endregion

		#pragma region PBC Logic
		if (r_0 < -d_sWidth / 2) {
			r_0 += d_sWidth;
			p.posMultiplier[0]--;
		}
		else if (r_0 >= d_sWidth / 2) {
			r_0 -= d_sWidth;
			p.posMultiplier[0]++;
		}

		if (r_1 < -d_sHeight / 2) {
			r_1 += d_sHeight;
			p.posMultiplier[1]--;
		}
		else if (r_1 >= d_sHeight / 2) {
			r_1 -= d_sHeight;
			p.posMultiplier[1]++;
		}
		#pragma endregion

		p.pos[0] = r_0;
		p.pos[1] = r_1;
		p.ort[0] = u_0;
		p.ort[1] = u_1;
	}
}

__device__ float InteractionForce(float r) {
	if (r < 1.122462f)
		return 48 * pow(1 / r, 13) - 24 * pow((1 / r), 7);
	else
		return 0;
}

float Distance(float dx, float dy)
{
	return sqrt(pow(dx, 2) + pow(dy, 2));
}

float DistancePBC(Matter m1, Matter m2, float &dx, float &dy) {
	dx = m1.pos[0] - m2.pos[0];
	dy = m1.pos[1] - m2.pos[1];

	#pragma region PBC Logic
	if (dx > scaledWindowWidth / 2)
		dx -= scaledWindowWidth;
	else if (dx <= -scaledWindowWidth / 2)
		dx += scaledWindowWidth;

	if (dy > scaledWindowHeight / 2)
		dy -= scaledWindowHeight;
	else if (dy <= -scaledWindowHeight / 2)
		dy += scaledWindowHeight;
	#pragma endregion

	return Distance(dx, dy);
}

float RadiansDifference(float radA, float radB) {
	radA += env_pi;
	radB += env_pi;
	float d = fmodf(abs(radA - radB), (float)env_pi*2);
	float r = d > env_pi ? env_pi*2 - d : d;
	if ((radA - radB >= 0 && radA - radB <= env_pi) || (radA - radB <= -env_pi && radA - radB >= -env_pi*2)) 
		return r;
	return -r;
}

float Environment::returnActiveWork() {
	float totalActiveWork = 0;
	for (int i = 0; i < matters.size(); i++) {
		Matter &p = matters[i];
		totalActiveWork += p.acmlActiveWork;
	}
	return (1 / ((float)(n+n_t)*t)) * totalActiveWork;
}

float Environment::returnCurrentActiveWork() {
	float totalActiveWork = 0;
	for (int i = 0; i < matters.size(); i++) {
		Matter &p = matters[i];
		totalActiveWork += p.acmlCurrentActiveWork;
	}
	return (1 / (float)(n + n_t)) * totalActiveWork;
}

void Environment::Display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	//draw matters
	for (int i = 0; i < matters.size(); i++) {
		Matter &p = matters[i];
		DrawMatter(p);

		#pragma region PBC Logic
		if (p.pos[0] - p.r / 2 < -scaledWindowWidth / 2) {
			DrawMatter(p, +scaledWindowWidth, 0);
		}
		else if (p.pos[0] + p.r / 2 >= scaledWindowWidth / 2) {
			DrawMatter(p, -scaledWindowWidth, 0);
		}

		if (p.pos[1] - p.r / 2 < -scaledWindowHeight / 2) {
			DrawMatter(p, 0, scaledWindowHeight);
		}
		else if (p.pos[1] + p.r / 2 >= scaledWindowHeight / 2) {
			DrawMatter(p, 0, -scaledWindowHeight);
		}
		#pragma endregion
	}

	glFlush();
	glutSwapBuffers();
}

void DrawMatter(Matter p, float transformX, float transformY) {
	glPushMatrix();
	glScalef(SCALE_FACTOR, SCALE_FACTOR, 1.0);
	glTranslatef(p.pos[0] + transformX, p.pos[1] + transformY, 0.0f);

	glColor3f(p.ort[0], p.ort[1], 1);

	glBegin(GL_POLYGON);
	for (float a = 0; a < 2 * env_pi; a += 0.2)
		glVertex2f(p.r / 2 * cos(a), p.r / 2 * sin(a));
	glEnd();

	float degree = -(atan2(p.ort[0], p.ort[1]) * 180 / env_pi);
	glRotatef(degree, 0.0f, 0.0f, 1.0f);

	glColor3f(0, 0, 0);
	glBegin(GL_TRIANGLES);
	glVertex2f(0, 0.45);
	glVertex2f(-0.25, -0.25);
	glVertex2f(0.25, -0.25);
	glEnd();

	glPopMatrix();
}

Vector2f RandomPos() {
	int randSignX = (rand() > RAND_MAX / 2) ? -1 : 1;
	float x = randSignX * (rand() / (RAND_MAX / (floor(windowWidth / SCALE_FACTOR) / 2)));
	int randSignY = (rand() > RAND_MAX / 2) ? -1 : 1;
	float y = randSignY * (rand() / (RAND_MAX / (floor(windowHeight / SCALE_FACTOR) / 2)));

	Vector2f pos(x, y);
	return pos;
}

Vector2f RandomOrt() {
	int randTotal = (rand() > RAND_MAX / 2) ? -1 : 1;
	float x = 2 * (float)rand() / (float)RAND_MAX - 1;

	float y = randTotal - x;
	Vector2f ort(x, y);
	return ort;
}

void Environment::AddMatter(MatterType mt)
{
	Matter p;
	Vector2f pos = RandomPos();
	p.pos[0] = pos(0);
	p.pos[1] = pos(1);
	p.v = V;
	p.r = RADIUS;
	Vector2f ort = RandomOrt();
	p.ort[0] = ort(0);
	p.ort[1] = ort(1);
	p.type = mt;
	matters.push_back(p);
	prevMatters.push_back(p);
}

void Environment::AddMatter(MatterType mt, float x, float y, float ortx, float orty)
{
	Matter p;
	p.pos[0] = x;
	p.pos[1] = y;
	p.v = V;
	p.r = RADIUS;
	p.ort[0] = ortx;
	p.ort[1] = orty;
	p.type = mt;
	matters.push_back(p);
	prevMatters.push_back(p);
}

void Environment::RemoveMatters()
{
	matters.clear();
	prevMatters.clear();
}

Environment::~Environment()
{
	RemoveMatters();
}