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
#include <ctime>

void trainTimer(int = 0);
void resultTimer(int = 0);
void display();
void keyboard(unsigned char, int, int);
void drawMatter(Matter, float = 0, float = 0);
float returnActiveWork();

extern float windowWidth;
extern float windowHeight;
extern const float SCALE_FACTOR;
extern float scaledWindowWidth;
extern float scaledWindowHeight;
extern float t;
extern float dt;
extern int n;

int currentEpisode = 1;
int totalEpisode = 1000;
float T = 100; // Total Timestep
bool isLearning = false;
bool displayEnabled = true;

int WIN;
int startTime = time(NULL);

Environment env = Environment(1, 32, false);
Agent agent = Agent(0.01, 1, 10,  0, 0.9);

std::vector<float> currentState = env.ReturnState();
std::vector<int> actionID(n);
std::vector<float> reward(n);
std::vector<float> newState(n);
std::vector<float> action(n);
std::vector<float> activeWork(n);

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
		glutKeyboardFunc(keyboard);

		if (isLearning) {
			trainTimer();
		} else {
			agent.LoadQTable("test.csv");
			agent.LoadDTable();
			agent.LoadSVTable();
			agent.LoadTPMatrix();
			agent.setEpsilon(0);
			resultTimer();
		}
		glutMainLoop();
	} else {
		while (currentEpisode <= totalEpisode) {
			int currentTime = time(NULL);
			env = Environment(1, 32);
			currentState = env.ReturnState();
			std::cout << "episode started: " << currentEpisode << std::endl;

			while (t < T)
			{
				action = agent.ReturnAction(currentState, actionID);
				newState = env.Step(action, reward, activeWork);
				agent.UpdateQTable(currentState, actionID, reward, newState);
				currentState = newState;

				if (agent.returnEpsilon() < 0.1) {
					std::cout << "Epsilon < 1 " << std::endl;
					std::cin.get();
				}
			}

			float normActiveWork = returnActiveWork();
			std::fill(activeWork.begin(), activeWork.end(), 0);
			agent.UpdateTPMatrix();
			agent.UpdateEpsilonDecay(currentEpisode, totalEpisode);
			std::cout << "episode ended: " << currentEpisode << std::endl;
			std::cout << "epsilon is : " << agent.returnEpsilon() << std::endl;
			std::cout << "active work is : " << normActiveWork << std::endl;
			std::cout << "seconds used : " << time(NULL) - currentTime << std::endl;
			std::cout << "------------------------" << std::endl;
			currentEpisode++;
			t = 0;
			agent.SaveQTable("test.csv");
			agent.SaveSVTable();
			agent.SaveDTable();
			agent.SaveTPMatrix();
		}
	}
	std::cin.get();
	return 0;
}

void trainTimer(int)
{
	display();

	action = agent.ReturnAction(currentState, actionID);
	newState = env.Step(action, reward, activeWork);
	agent.UpdateQTable(currentState, actionID, reward, newState);
	agent.UpdateEpsilonDecay(t, T*100);
	currentState = newState;

	glutTimerFunc(1, trainTimer, 0);
}

void resultTimer(int) {
	display();

	action = agent.ReturnAction(currentState, actionID);
	newState = env.Step(action, reward, activeWork);
	currentState = newState;

	glutTimerFunc(1, resultTimer, 0);
}

float returnActiveWork() {
	float totalActiveWork = 0;
	for each (float work in activeWork)
		totalActiveWork += work;
	return (1 / (n*t)) * totalActiveWork;
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
		if (p.pos[0] - p.r / 2 < -scaledWindowWidth / 2)
			drawMatter(p, +scaledWindowWidth, 0);
		else if (p.pos[0] + p.r / 2 >= scaledWindowWidth / 2)
			drawMatter(p, -scaledWindowWidth, 0);

		if (p.pos[1] - p.r / 2 < -scaledWindowHeight / 2)
			drawMatter(p, 0, scaledWindowHeight);
		else if (p.pos[1] + p.r / 2 >= scaledWindowHeight / 2)
			drawMatter(p, 0, -scaledWindowHeight);	
		#pragma endregion
	}

	glFlush();
	glutSwapBuffers();
}

void drawMatter(Matter p, float transformX, float transformY) {
	glPushMatrix();
	glScalef(SCALE_FACTOR, SCALE_FACTOR, 1.0);
	glTranslatef(p.pos[0] + transformX, p.pos[1] + transformY, 0.0f);
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

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 27: {
			// "esc"
			glutDestroyWindow(WIN);
			exit(0);
		}
		case 49: {
			// "1"
			std::cout << "Print Q Table : " << t << std::endl;
			for (int i = 0; i < 21; ++i)
			{
				for (int j = 0; j < 21; ++j)
				{
					std::cout << agent.QTable[i][j] << '\t';
				}
				std::cout << std::endl << "----------------------------" << std::endl;
			}
			agent.SaveQTable("test.csv");
			agent.SaveSVTable();
			std::cin.get();
			break;
		}
		case 50: {
			// "2"
			std::cout << "Print agent status:" << std::endl;
			std::cout << "----------------------------" << std::endl;
			std::cout << "current epsilon: " << agent.returnEpsilon() << std::endl;
			std::cout << "current active work : " << returnActiveWork() << std::endl;
			std::cout << "seconds passed: " << std::to_string(time(NULL) - startTime) << std::endl;
			std::cout << "----------------------------" << std::endl;
			break;
		}
		case 51: {
			// "3"
			std::cout << "Print time : " << t << std::endl;
			std::cin.get();
			break;
		}
		case 52: {
			// "4"
			std::cout << "Loading Q Table from CSV..." << std::endl;
			agent.LoadQTable("test.csv");
			break;
		}
	}
}