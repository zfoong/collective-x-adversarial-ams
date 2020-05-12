#include <iostream>
#include "display.h"
#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

struct Matter {
	float x; // x pos
	float y; // y pos
	float r; // radius, setting to const for now
	float v;
	float m; // mass of matter
	float ort[2];
	float color[3];
};

void timer(int = 0);
void display();
void mouse(int, int, int, int);
void addMatter(float, float, float = 1);
void removeMatters();
void keyboard(unsigned char, int, int);
void movement(Matter&);
void movement_dep(Matter&);
float clipToScreen(float);

float windowWidth = 500;
float windowHeight = 500;

int Mx, My, WIN;
bool PRESSED_LEFT = false;
const int RADIUS = 10;
const int MASS = 100;

int t = 0; // current time
float dt = 1; // time step
int n = 1; // total amt of matter

float transDif = 0.0055f;
float rotDif = 0.0166f;
float transDifCoef = sqrt(2 * transDif);
float rotDifCoef = sqrt(2 * rotDif);

std::vector<Matter> prevMatters;
std::vector<Matter> matters;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0, 1);

int main(int argc, char **argv)
{
	Matter mtest;
	// spawn test
	mtest.x = 0;
	mtest.y = 0;
	mtest.v = 1;
	mtest.m = MASS;
	mtest.r = RADIUS;
	mtest.ort[0] = 1;
	mtest.ort[1] = 1;
	mtest.color[0] = 1; // TODO: changes for orietation
	mtest.color[1] = 0;
	mtest.color[2] = 0;
	matters.push_back(mtest);

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
		addMatter(MASS, RADIUS);
		PRESSED_LEFT = false;
	}

	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];
		movement(p);
	}

	t += dt;

	glutTimerFunc(1, timer, 0);
}

void movement(Matter &p) {
	float eta = (float)(distribution(generator));
	float xi_1 = (float)(distribution(generator));
	float xi_2 = (float)(distribution(generator));

	Vector2f r(p.x, p.y);
	Vector2f v(p.ort[0], p.ort[1]);
	Vector2f tranNoise(xi_1, xi_2);

	float theta = rotDifCoef * eta;

	Matrix2f R;
	R << cos(theta), -sin(theta),
		 sin(theta),  cos(theta);

	Vector2f u = R*v;

	r = r + p.v * u + transDifCoef * tranNoise;

	p.x = clipToScreen(r(0));
	p.y = clipToScreen(r(1));
	p.ort[0] = u(0);
	p.ort[1] = u(1);
}

void movement_dep(Matter &p) {
	float theta = (float)(distribution(generator));
	// theta = theta * 360;
	Vector2f r(p.x, p.y);
	Matrix2f R;
	R << cos(theta), -sin(theta),
		 sin(theta),  cos(theta);
	Vector2f u(p.ort[0], p.ort[1]);
	Vector2f v = R*u;
	r = r + 0.5 * v * dt;
	p.x = r(0);
	p.y = r(1);
	p.ort[0] = v(0);
	p.ort[1] = v(1);
}

float clipToScreen(float i) {
	return std::min( windowWidth/2 , std::max(i, -(windowWidth/2)));
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	//draw matters
	for (int i = 0; i < matters.size(); i++)
	{
		Matter &p = matters[i];

		glPushMatrix();
		glTranslatef(p.x, p.y, 0.0f);

		glColor3f(p.color[0], p.color[1], p.color[2]);
		glBegin(GL_POLYGON);
		for (float a = 0; a < 2 * M_PI; a += 0.2)
			glVertex2f(p.r*cos(a), p.r*sin(a));
		glEnd();

		float degree = -(atan2(p.ort[0], p.ort[1]) * 180 / M_PI);
		glRotatef(degree, 0.0f, 0.0f, 1.0f);

		glColor3f(0, 0, 0);
		glBegin(GL_TRIANGLES);
		glVertex2f(0, 7);
		glVertex2f(-5, -5);
		glVertex2f(5, -5);
		glEnd();

		glPopMatrix();
	}

	glFlush();
	glutSwapBuffers();
}

void mouse(int button, int state, int x, int y)
{
	//set the coordinates of cursor
	Mx = x - windowWidth/2;
	My = -(y - windowHeight/2);

	//check which button is pressed
	if (button == GLUT_LEFT_BUTTON)
		PRESSED_LEFT = state == GLUT_DOWN;
}

void addMatter(float m, float r, float v)
{
	Matter p;
	p.x = Mx;
	p.y = My;
	p.v = v;
	p.m = m;
	p.r = r;
	p.ort[0] = 1;
	p.ort[1] = 1;
	p.color[0] = 1;
	p.color[1] = 1;
	p.color[2] = 0;
	matters.push_back(p);
}

void removeMatters()
{
	for (int i = 0; i < matters.size(); i++)
		matters.pop_back();
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