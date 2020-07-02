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
#include <direct.h>

void trainTimer(int = 0);
void resultTimer(int = 0);
void display();
void keyboard(unsigned char, int, int);
void drawMatter(Matter, float = 0, float = 0);
std::string initSimulationData(int);
void saveSimulationData(std::string, int);
void updateSimulationResult(std::string, float);

extern float windowWidth;
extern float windowHeight;
extern const float SCALE_FACTOR;
extern float scaledWindowWidth;
extern float scaledWindowHeight;
extern float t;
extern float dt;
extern int n;
extern int n_t;

int currentEpisode = 1;
int totalEpisode = 1000;
float T = 100; // Total Timestep
bool isLearning = true;
bool displayEnabled = false;

int WIN;
int startTime = time(NULL);

Environment env = Environment(81, 0, true);
Agent agent = Agent(0.05, 1, 10,  0, 0.9);

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
		glutKeyboardFunc(keyboard);

		if (isLearning) {
			trainTimer();
		} else {
			agent.LoadQTable("QTable");
			agent.LoadDTable("DTable");
			agent.LoadSVTable("SVTable");
			agent.LoadTPMatrix("TPMatrix");
			agent.setEpsilon(0);
			agent.SortStateValueList();
			resultTimer();
		}
		glutMainLoop();
	} else {
		std::string dir = initSimulationData(startTime);
		while (currentEpisode <= totalEpisode) {
			int currentTime = time(NULL);
			env = Environment(81, 0);
			currentState = env.ReturnState();
			std::cout << "current epsilon is : " << agent.returnEpsilon() << std::endl;
			std::cout << "current learning rate is : " << agent.returnLearningRate() << std::endl;
			std::cout << "episode started: " << currentEpisode << std::endl;

			while (t < T)
			{
				action = agent.ReturnAction(currentState, actionID);
				newState = env.Step(action, reward);
				agent.UpdateQTable(currentState, actionID, reward, newState);
				currentState = newState;

				/*if (agent.returnEpsilon() < 0.1) {
					std::cout << "Epsilon < 1 " << std::endl;
					std::cin.get();
				}*/
			}

			float normActiveWork = env.returnActiveWork();
			agent.SortStateValueList();
			agent.UpdateTPMatrix();
			agent.UpdateEpsilonDecay(currentEpisode, totalEpisode);
			agent.UpdateLearningRateDecay(currentEpisode, totalEpisode);
			std::cout << "episode ended: " << currentEpisode << std::endl;
			std::cout << "seconds used : " << time(NULL) - currentTime << std::endl;
			std::cout << "active work is : " << normActiveWork << std::endl;
			std::cout << "------------------------" << std::endl;
			saveSimulationData(dir, currentEpisode);
			updateSimulationResult(dir, normActiveWork);
			currentEpisode++;
			t = 0;
		}
	}
	std::cin.get();
	return 0;
}

void trainTimer(int)
{
	display();

	action = agent.ReturnAction(currentState, actionID);
	newState = env.Step(action, reward);
	agent.UpdateQTable(currentState, actionID, reward, newState);
	agent.UpdateEpsilonDecay(t, T*100);
	currentState = newState;
	glutTimerFunc(1, trainTimer, 0);
}

void resultTimer(int) {
	display();

	action = agent.ReturnAction(currentState, actionID);
	newState = env.Step(action, reward);

	std::cout << "state is " << newState[0] << std::endl;
	std::cout << "action is " << action[0] << std::endl;
	std::cin.get();

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

std::string initSimulationData(int time) {
	int check;
	std::string dir = "data\\";
	std::string dirHead = "data_";
	std::string propFileName = "properties.txt";
	dir = dir + dirHead + std::to_string(time);
	check = _mkdir(dir.c_str());

	if (!check)
		std::cout << "Directory created: " << dir << std::endl;
	else {
		std::cout << "Failed to created directory: " << dir << std::endl;
		exit(1);
	}

	std::ofstream outFile(dir + "\\" + propFileName);
	outFile << "time step = " << dt << '\n';
	outFile << "total time step = " << T << '\n';
	outFile << "total episode = " << totalEpisode << '\n';
	outFile << "learning rate = " << agent.learningRate << '\n';
	outFile << "learning rate decay = " << agent.learningRateDecay << '\n';
	outFile << "epsilon = " << agent.epsilon << '\n';
	outFile << "epsilon decay = " << agent.epsilonDecay << '\n';
	outFile << "discount factor = " << agent.discountFactor << '\n';
	outFile << "noise = " << '\n';
	outFile << "range = " << '\n';
	outFile << "density = " << '\n';
	outFile << "learner count = " << n << '\n';
	outFile << "teacher count = " << n_t << '\n';
	outFile.close();

	return dir;
}

void saveSimulationData(std::string dir, int episode) {
	int check = 0;
	std::string subDir = "ep_";
	dir = dir + "\\" + subDir + std::to_string(episode);
	check = _mkdir(dir.c_str());

	if (!check)
		std::cout << "Directory created: " << dir << std::endl;
	else {
		std::cout << "Failed to created directory: " << dir << std::endl;
		exit(1);
	}

	agent.SaveQTable((dir + "\\" + "QTable").c_str());
	agent.SaveSVTable((dir + "\\" + "SVTable").c_str());
	agent.SaveDTable((dir + "\\" + "DTable").c_str());
	agent.SaveTPMatrix((dir + "\\" + "TPMatrix").c_str());
}

void updateSimulationResult(std::string dir, float activeWork) {
	std::string resultFileName = "result.csv";
	dir = dir + "\\" + resultFileName;
	//if (!std::ifstream(dir))
	//{
	//	int check = _mkdir(dir.c_str());
	//	if (!check)
	//		std::cout << "Directory created: " << dir << std::endl;
	//	else {
	//		std::cout << "Failed to created directory: " << dir << std::endl;
	//		exit(1);
	//	}
	//}
	std::ofstream outFile(dir, std::ofstream::app);
	outFile << activeWork << '\n';
	outFile.close();
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
			agent.SaveQTable("QTable");
			agent.SaveSVTable("SVTable");
			std::cin.get();
			break;
		}
		case 50: {
			// "2"
			std::cout << "Print agent status:" << std::endl;
			std::cout << "----------------------------" << std::endl;
			std::cout << "current epsilon: " << agent.returnEpsilon() << std::endl;
			std::cout << "current active work : " << env.returnActiveWork() << std::endl;
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
			agent.LoadQTable("QTable");
			break;
		}
	}
}