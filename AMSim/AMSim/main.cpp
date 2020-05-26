#include <iostream>
#include "display.h"
#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

const float RADIUS = 1;
const float MASS = 100;
const float V = 1;

enum MatterType {
	collective,
	adversarial
};

struct Matter {
	float x; // x pos
	float y; // y pos
	float r = RADIUS; // radius, setting to const for now
	float v = V;
	float m = MASS; // mass of matter
	float ort[2];
	float totalDis = 0;
	MatterType type = collective;
};

void timer(int = 0);
void display();
void drawMatter(Matter, float = 0, float = 0);
void mouse(int, int, int, int);
void addMatter(float, float);
void addMatterByMouse(float, float);
void removeMatters();
void keyboard(unsigned char, int, int);
void movement(Matter&);
void movement_dep(Matter&);
float clipToScreen(float);
float interactionForce(float);
float distance(float, float);
Vector2f randomOrt();

float windowWidth = 500;
float windowHeight = 500;

int Mx, My, WIN;
const float SCALE_FACTOR = 20;
float scaledWindowWidth = windowWidth / SCALE_FACTOR;
float scaledWindowHeight = windowHeight / SCALE_FACTOR;
bool PRESSED_LEFT = false;

float t = 0; // current time
float dt = 0.01f; // time step
int n = 64; // total amt of matter, set to 4,9,16,25,36,49,64,81,100

float pl = 60;
float rotDif = V/pl;
float transDif = pow(RADIUS, 2) * rotDif / 3;
float rotDifCoef = sqrt(2 * rotDif);
float transDifCoef = sqrt(2 * transDif);
float alpha = transDif / rotDif * pow(RADIUS, 2);
float mu = alpha * RADIUS / pl;

std::vector<Matter> prevMatters;
std::vector<Matter> matters;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0, 1);

int main(int argc, char **argv)
{
	for (int i = 0; i < sqrt(n); i++) {
		for (int j = 0; j < sqrt(n); j++) {
			addMatter(i * 2  - (windowWidth / 2 / SCALE_FACTOR / 2 ), j * 2 - (windowHeight / 2 / SCALE_FACTOR/ 2 ));
		}
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(50, 50);
	WIN = glutCreateWindow("AMSim");

	glClearColor(0, 0, 0, 1);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-(windowWidth / 2), (windowWidth / 2), -(windowHeight / 2), (windowHeight / 2), 0.0f, 1.0f);

	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);
	timer();

	glutMainLoop();
	return 0;
}

void timer(int)
{
	display();
	if (PRESSED_LEFT)
	{
		addMatterByMouse(MASS, RADIUS);
		PRESSED_LEFT = false;
	}

	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];
		movement(p);
	}

	t += dt;

	//if (t >= 10) {
	//	// calculate total dis
	//	float totalDis = 0;
	//	for (int i = 0; i < matters.size(); i++)
	//		totalDis += matters[i].totalDis;
	//	float totalDisNorm = totalDis / (n*t);
	//	std::cout << totalDisNorm << std::endl;
	//	// std::cin.get();
	//}

	prevMatters = matters;
	glutTimerFunc(1, timer, 0);
}

void movement(Matter &p) {
	float eta = (float)(distribution(generator));
	float xi_1 = (float)(distribution(generator));
	float xi_2 = (float)(distribution(generator));

	Vector2f r(p.x, p.y);
	Vector2f rPrev(p.x, p.y);
	Vector2f tranNoise(xi_1, xi_2);

	float rad = atan2(p.ort[1], p.ort[0]); // convert ort vector to radians
	float theta = rad + sqrt(dt) * (rotDifCoef * eta); 
	Vector2f u(cos(theta), sin(theta)); //convert theta to ort vector u
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

		float dis = distance(dx, dy);
		
		if (dis == 0)
			continue;

		float Fr = interactionForce(dis);

		float forceAngle = atan2(dy, dx);

		F(0) += Fr * cos(forceAngle);
		F(1) += Fr * sin(forceAngle);
	}
	
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

	//p.totalDis += distance(p.x, p.y, rPrev(0), rPrev(1));
}

float interactionForce(float r) {
	if (r < 1.122462f)
		return 48 * pow(1/r, 13) - 24 * pow((1/r), 7);
	else
		return 0;
}

float clipToScreen(float i) {
	return std::min( (windowWidth/2) / SCALE_FACTOR , std::max(i, -(windowWidth/2) / SCALE_FACTOR));
}

float distance(float dx, float dy)
{
	return sqrt(pow(dx, 2) + pow(dy, 2));
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	//draw matters
	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];
		drawMatter(p);

		#pragma region PBC Logic
		if (p.x - p.r / 2 < -scaledWindowWidth / 2) {
			drawMatter(p, +scaledWindowWidth, 0);
		}
		else if (p.x + p.r / 2 >= scaledWindowWidth / 2) {
			drawMatter(p, -scaledWindowWidth, 0);
		}

		if (p.y - p.r / 2 < -scaledWindowHeight / 2) {
			drawMatter(p, 0, scaledWindowHeight);
		}
		else if (p.y + p.r / 2 >= scaledWindowHeight / 2) {
			drawMatter(p, 0, -scaledWindowHeight);
		}	
		#pragma endregion
	}

	glFlush();
	glutSwapBuffers();
}

void drawMatter(Matter p, float transformX, float transformY) {
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

void mouse(int button, int state, int x, int y)
{
	//set the coordinates of cursor
	Mx = (x - windowWidth/2) / SCALE_FACTOR;
	My = -(y - windowHeight/2) / SCALE_FACTOR;

	//check which button is pressed
	if (button == GLUT_LEFT_BUTTON)
		PRESSED_LEFT = state == GLUT_DOWN;
}

Vector2f randomOrt() {
	int randTotal = (rand() > RAND_MAX / 2) ? -1 : 1;
	float x = 2 * (float)rand() / (float)RAND_MAX - 1;
	
	float y = randTotal - x;
	Vector2f ort(x, y);
	return ort;
}

void addMatterByMouse(float m, float r)
{
	Matter p;
	p.x = Mx;
	p.y = My;
	p.v = V;
	p.m = m;
	p.r = r;
	Vector2f ort = randomOrt();
	p.ort[0] = ort(0);
	p.ort[1] = ort(1);
	matters.push_back(p);
	prevMatters.push_back(p);
}

void addMatter(float x, float y)
{
	Matter p;
	p.x = x;
	p.y = y;
	p.v = V;
	p.m = MASS;
	p.r = RADIUS;
	Vector2f ort = randomOrt();
	p.ort[0] = ort(0);
	p.ort[1] = ort(1);
	matters.push_back(p);
	prevMatters.push_back(p);
}

void removeMatters()
{
	matters.clear();
	prevMatters.clear();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		removeMatters();
		glutDestroyWindow(WIN);
		exit(0);
		break;
	}
}