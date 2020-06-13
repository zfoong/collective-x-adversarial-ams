#include "environment.h"
#include <iostream>
#include "display.h"
#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

float windowWidth = 500;
float windowHeight = 500;
extern const float SCALE_FACTOR = 20;
float scaledWindowWidth = windowWidth / SCALE_FACTOR;
float scaledWindowHeight = windowHeight / SCALE_FACTOR;

float t = 0; // current time
float dt = 0.01f; // time step
int n = 1; // total amt of matter, set to 4,9,16,25,36,49,64,81,100

float pl = 60;
float rotDif = V / pl;
float transDif = pow(RADIUS, 2) * rotDif / 3;
float rotDifCoef = sqrt(2 * rotDif);
float transDifCoef = sqrt(2 * transDif);
float alpha = transDif / rotDif * pow(RADIUS, 2);
float mu = alpha * RADIUS / pl;
float range = 5;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0, 1);

float InteractionForce(float);
float Distance(float, float);
float RadiansDifference(float, float);
void DrawMatter(Matter, float = 0, float = 0);
Vector2f RandomOrt();
Vector2f RandomPos();

Environment::Environment(float num)
{
	n = num;
	int teacherCount = 32;
	for (int i = 0; i < sqrt(teacherCount); i++) {
		for (int j = 0; j < sqrt(teacherCount); j++) {
			Vector2f ort = RandomOrt();
			AddMatter(teacher, i * 3 - (windowWidth / 2 / SCALE_FACTOR / 2), j * 3 - (windowHeight / 2 / SCALE_FACTOR / 2), ort(0), ort(1));
		}
	}
	/*AddMatter(teacher, -1, 0, 0, 1);
	AddMatter(teacher, 0, 1, 0, 1);
	AddMatter(teacher, 1, 0, 0, 1);
	AddMatter(learner, 0, 0, 1, 0);*/
	for (int i = 0; i < n; i++) {
		AddMatter(learner, -5, -5, 0.5, 0.5);
	}
	//AddMatter(learner, 3, 0, 0, -1);
}

std::vector<float> Environment::ReturnState()
{
	std::vector<float> state;
	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];
		if (p.type == teacher) continue;

		int inRangeCount = 0;
		float totalRad = 0;
		float ortx = 0;
		float orty = 0;
		for (int j = 0; j < prevMatters.size(); j++)
		{
			Matter &m = prevMatters[j];
			float dx = p.x - m.x;
			float dy = p.y - m.y;

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

			float dis = Distance(dx, dy);

			if (dis == 0)
				continue;

			if (dis < range) {
				ortx += m.ort[0];
				orty += m.ort[1];
				inRangeCount++;
			}
		}
		float radDiff = 0;
		if (inRangeCount != 0) {
			// TODO: find difference logic fix
			// float radDiff = atan2(orty, ortx) - atan2(p.ort[1], p.ort[0]);
			radDiff = RadiansDifference(atan2(orty, ortx), atan2(p.ort[1], p.ort[0]));
		}
		state.push_back(radDiff);
	}
	return state;
}

std::vector<float> Environment::Step(std::vector<float> actionList, std::vector<float> &rewardList) 
{
	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];
		float a = 0;
		if(p.type == learner) a = actionList[0];
		Movement(p, a);
	}

	t += dt;
	prevMatters = matters;

	// calc reward by neighbour lost
	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];
		if (p.type == teacher) continue;

		int inRangeCount = 0;

		for (int j = 0; j < prevMatters.size(); j++)
		{
			Matter &m = prevMatters[j];
			float dx = p.x - m.x;
			float dy = p.y - m.y;

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

			float dis = Distance(dx, dy);

			if (dis == 0)
				continue;

			if (dis < range)
				inRangeCount++;
		}
		if (inRangeCount < p.neighbourCount)
			rewardList[0] = inRangeCount - p.neighbourCount;
		else
			rewardList[0] = 0;
		p.neighbourCount = inRangeCount;
	}
	return ReturnState();
}

void Environment::Movement(Matter &p, float action) {
	float eta = (float)(distribution(generator));
	float xi_1 = (float)(distribution(generator));
	float xi_2 = (float)(distribution(generator));

	float avgOrt = 0;
	float ortx = 0;
	float orty = 0;

	Vector2f r(p.x, p.y);
	Vector2f rPrev(p.x, p.y);
	Vector2f tranNoise(xi_1, xi_2);
	Vector2f F(0, 0);

	for (int i = 0; i < prevMatters.size(); i++)
	{
		Matter &m = prevMatters[i];

		float dx = p.x - m.x;
		float dy = p.y - m.y;

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

		float dis = Distance(dx, dy);

		if (dis == 0)
			continue;

		if (dis < range && m.type != learner) {
			ortx += m.ort[0];
			orty += m.ort[1];
		}

		float Fr = InteractionForce(dis);
		float forceAngle = atan2(dy, dx);
		F(0) += Fr * cos(forceAngle);
		F(1) += Fr * sin(forceAngle);
	}
	float rad = atan2(p.ort[1], p.ort[0]); // convert ort vector to radians
	float theta = rad + -action * dt + sqrt(dt) * (rotDifCoef * eta);

	if (p.type == teacher) {
		float radDiff = RadiansDifference(atan2(orty, ortx), rad);
		if (ortx == 0 && orty == 0)
			radDiff = 0;
		theta = rad + radDiff * dt + sqrt(dt) * (rotDifCoef * eta);
	}

	Vector2f u(cos(theta), sin(theta)); // convert theta to ort vector u

	r = r + dt * (mu * F) + dt * (p.v * u) + sqrt(dt) * (transDifCoef * tranNoise);

	#pragma region PBC Logic
	if (r(0) < -scaledWindowWidth / 2)
		r(0) += scaledWindowWidth;
	else if (r(0) >= scaledWindowWidth / 2)
		r(0) -= scaledWindowWidth;

	if (r(1) < -scaledWindowHeight / 2)
		r(1) += scaledWindowHeight;
	else if (r(1) >= scaledWindowHeight / 2)
		r(1) -= scaledWindowHeight;
	#pragma endregion

	p.x = r(0);
	p.y = r(1);
	p.ort[0] = u(0);
	p.ort[1] = u(1);
}

float InteractionForce(float r) {
	if (r < 1.122462f)
		return 48 * pow(1 / r, 13) - 24 * pow((1 / r), 7);
	else
		return 0;
}

float Distance(float dx, float dy)
{
	return sqrt(pow(dx, 2) + pow(dy, 2));
}

float RadiansDifference(float radA, float radB) {
	radA += M_PI;
	radB += M_PI;
	float d = fmodf(abs(radA - radB), (float)M_PI*2);
	float r = d > M_PI ? M_PI*2 - d : d;
	if ((radA - radB >= 0 && radA - radB <= M_PI) || (radA - radB <= -M_PI && radA - radB >= -M_PI*2)) return r;
	return -r;
}

void Environment::Display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	//draw matters
	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];
		DrawMatter(p);

		#pragma region PBC Logic
		if (p.x - p.r / 2 < -scaledWindowWidth / 2) {
			DrawMatter(p, +scaledWindowWidth, 0);
		}
		else if (p.x + p.r / 2 >= scaledWindowWidth / 2) {
			DrawMatter(p, -scaledWindowWidth, 0);
		}

		if (p.y - p.r / 2 < -scaledWindowHeight / 2) {
			DrawMatter(p, 0, scaledWindowHeight);
		}
		else if (p.y + p.r / 2 >= scaledWindowHeight / 2) {
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
	glTranslatef(p.x + transformX, p.y + transformY, 0.0f);

	glColor3f(p.ort[0], p.ort[1], 1);

	glBegin(GL_POLYGON);
	for (float a = 0; a < 2 * M_PI; a += 0.2)
		glVertex2f(p.r / 2 * cos(a), p.r / 2 * sin(a));
	glEnd();

	float degree = -(atan2(p.ort[0], p.ort[1]) * 180 / M_PI);
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
	p.x = pos(0);
	p.y = pos(1);
	p.v = V;
	p.m = MASS;
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
	p.x = x;
	p.y = y;
	p.v = V;
	p.m = MASS;
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

}