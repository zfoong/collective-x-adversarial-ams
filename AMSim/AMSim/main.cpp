#define _USE_MATH_DEFINES

#include "environment.h"
#include "agent.h"
#include <iostream>
#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <fstream>

void trainTimer(int = 0);
void resultTimer(int = 0);
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
extern float dt;
extern int n;

int currentEpisode = 1;
int totalEpisode = 10;
bool isLearning = false;
bool displayEnabled = true;
float T = 1000;
int Mx, My, WIN;
bool PRESSED_LEFT = false;

Environment env = Environment(1);
Agent agent = Agent();
std::vector<float> currentState = env.ReturnState();
std::vector<int> actionID(n);
std::vector<float> reward(n);
std::vector<float> newState(n);
std::vector<float> action(n);

int main(int argc, char **argv)
{
	if (displayEnabled) {
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

		if (isLearning) {
			trainTimer();
			glutMainLoop();
		} else {
			agent.LoadQTable();
			agent.setEpsilon(0);
			resultTimer();
			glutMainLoop();
		}

	} else {
		while (currentEpisode <= totalEpisode) {
			env = Environment(1);
			currentState = env.ReturnState();
			std::cout << "episode started: " << currentEpisode << std::endl;
			while (t < T)
			{
				action = agent.ReturnAction(currentState, actionID);
				newState = env.Step(action, reward);
				agent.UpdateQTable(currentState, actionID, reward, newState);
				currentState = newState;
			}
			agent.UpdateEpsilonDecay(t, T);
			std::cout << "episode ended: " << currentEpisode << std::endl;
			currentEpisode++;
			t = 0;
			agent.SaveQTable();
		}
	}
	return 0;
}

void trainTimer(int)
{
	display();
	action = agent.ReturnAction(currentState, actionID);
	newState = env.Step(action, reward);
	agent.UpdateQTable(currentState, actionID, reward, newState);
	currentState = newState;
	glutTimerFunc(1, trainTimer, 0);
}

void resultTimer(int) {
	display();
	action = agent.ReturnAction(currentState, actionID);
	newState = env.Step(action, reward);
	currentState = newState;
	glutTimerFunc(1, resultTimer, 0);
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
	if (p.type == learner)
		glColor3f(255, 255, 0);
	else
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
		case 27: {
			// "esc"
			// removeMatters();
			glutDestroyWindow(WIN);
			exit(0);
		}
		case 49: {
			// "1"
			std::cout << "Print Q Table : " << t << std::endl;
			for (int i = 0; i < 16; ++i)
			{
				for (int j = 0; j < 16; ++j)
				{
					std::cout << agent.QTable[i][j] << '\t';
				}
				std::cout << std::endl << "----------------------------" << std::endl;
			}
			//agent.SaveQTable();
			std::cin.get();
			break;
		}
		case 50: {
			// "2"
			std::cout << "Print agent status:" << std::endl;
			std::cout << "----------------------------" << std::endl;
			std::cout << agent.PrintEpsilon() << std::endl;
			std::cout << "----------------------------" << std::endl;
			break;
		}
		case 51: {
			// "3"
			std::cout << "Print time : " << t << std::endl;
			break;
		}
		case 52: {
			// "4"
			std::cout << "Loading Q Table from CSV..." << std::endl;
			agent.LoadQTable();
			break;
		}
	}
}