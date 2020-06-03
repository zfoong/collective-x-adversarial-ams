#define _USE_MATH_DEFINES

#include <iostream>
#include "environment.h"
#include "agent.h"
#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

void timer(int = 0);
void display();
void mouse(int, int, int, int);
void keyboard(unsigned char, int, int);
void drawMatter(Matter, float = 0, float = 0);

extern float windowWidth;
extern float windowHeight;
extern const float SCALE_FACTOR;
extern float scaledWindowWidth;
extern float scaledWindowHeight;
extern float t;
extern int n;

int Mx, My, WIN;
bool PRESSED_LEFT = false;

Environment env = Environment();
Agent agent = Agent();
std::vector<float> currentState = env.ReturnState();

int main(int argc, char **argv)
{
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

	// testing site
	/*std::cout << "old!" << std::endl;
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			std::cout << agent.QTable[i][j] << ' ';
		}
		std::cout << std::endl;
	}*/


	std::vector<int> actionID(n);
	std::vector<float> reward(n);
	std::vector<float> action = agent.ReturnAction(currentState, actionID);
	std::vector<float> newState = env.Step(action, reward);
	agent.UpdateQTable(currentState, actionID, reward, newState);
	currentState = newState;

	glutTimerFunc(1, timer, 0);
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	//draw matters
	for (int i = 0; i < env.matters.size(); i++)
	{
		Matter &p = env.matters[i];
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

//void addMatterByMouse(float m, float r)
//{
//	Matter p;
//	p.x = Mx;
//	p.y = My;
//	p.v = V;
//	p.m = m;
//	p.r = r;
//	Vector2f ort = randomOrt();
//	p.ort[0] = ort(0);
//	p.ort[1] = ort(1);
//	matters.push_back(p);
//	prevMatters.push_back(p);
//}
//
//void addMatter(float x, float y)
//{
//	Matter p;
//	p.x = x;
//	p.y = y;
//	p.v = V;
//	p.m = MASS;
//	p.r = RADIUS;
//	Vector2f ort = randomOrt();
//	p.ort[0] = ort(0);
//	p.ort[1] = ort(1);
//	matters.push_back(p);
//	prevMatters.push_back(p);
//}
//
//void removeMatters()
//{
//	matters.clear();
//	prevMatters.clear();
//}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		// removeMatters();
		glutDestroyWindow(WIN);
		exit(0);
	}
}