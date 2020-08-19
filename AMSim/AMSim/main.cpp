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
#include <cuda.h>
#include <thread>

void trainTimer(int = 0);
void resultTimer(int = 0);
void display();
void keyboard(unsigned char, int, int);
void drawMatter(Matter, float = 0, float = 0);
std::string initSimulationData(int);
std::string createSimulationDir(std::string, int);
void saveEpisodeData(std::string, int, Agent, Agent);
void updateSimulationResult(std::string, std::string, float);
void updateEpsilon(std::string, float);
void readConfig();

extern float windowWidth;
extern float windowHeight;
extern const float SCALE_FACTOR;
extern float scaledWindowWidth;
extern float scaledWindowHeight;
//extern float t;
extern float dt;
extern int n_a;
extern int n_c;

int currentSimulation = 1;
int totalSimulation = 10;
int currentEpisode_1 = 1;
int totalEpisode = 120;
float T = 100; // Total Timestep
bool terminate_t = false;
bool isLearning = true;
bool displayEnabled = false;

int WIN;
int startTime = time(NULL);

Environment env = Environment(25, 10, true);
Agent agent_c = Agent(0.1, 0.2, 10, 0, 0.8);
Agent agent_a = Agent(0.1, 0.2, 10, 0, 0.8);

std::vector<float> currentState = env.ReturnAllState();
int n = n_a + n_c;
std::vector<int> actionID(n);
std::vector<float> reward(n);

int bufferSize = 10000;

void training(std::string dir, int sim_id) {
	std::string simDir = createSimulationDir(dir, sim_id);
	Agent agent_c = Agent(0.1, 0.2, 10, 0, 0.8);
	Agent agent_a = Agent(0.1, 0.2, 10, 0, 0.8);

	agent_c.LoadDTable("DTable_c");
	agent_c.LoadSVTable("SVTable_c");
	agent_c.LoadTPMatrix("TPMatrix_c");
	agent_c.SortStateValueList();

	agent_a.LoadDTable("DTable_a");
	agent_a.LoadSVTable("SVTable_a");
	agent_a.LoadTPMatrix("TPMatrix_a");
	agent_a.SortStateValueList();

	std::cout << "------------------------" << std::endl;
	std::cout << "Simulation started: " << sim_id << std::endl;
	int currentEpisode = 1;

	while (currentEpisode <= totalEpisode) {
		Environment env_ = Environment(25, 10);
		std::vector<float> currentState = env_.ReturnAllState();
		std::vector<int> actionID(n);
		std::vector<float> reward(n);
		std::vector<float> newState(n);
		std::vector<float> action(n);

		std::vector<std::vector<float>> currentStateList_a(n_a);
		std::vector<std::vector<int>> actionIDList_a(n_a);
		std::vector<std::vector<float>> rewardList_a(n_a);
		std::vector<std::vector<float>> newStateList_a(n_a);

		std::vector<std::vector<float>> currentStateList_c(n_c);
		std::vector<std::vector<int>> actionIDList_c(n_c);
		std::vector<std::vector<float>> rewardList_c(n_c);
		std::vector<std::vector<float>> newStateList_c(n_c);

		int currentTime = time(NULL);
		terminate_t = false;
		std::cout << "current epsilon is : " << agent_c.returnEpsilon() << std::endl;
		std::cout << "current learning rate is : " << agent_c.returnLearningRate() << std::endl;
		std::cout << "episode started: " << currentEpisode << std::endl;

		while (env_.t < T)
		{
			std::vector<float> currentState_a(currentState.begin(), currentState.begin() + n_a);
			std::vector<float> currentState_c(currentState.begin() + n_a, currentState.end());
			std::vector<int> actionID_a(actionID.begin(), actionID.begin() + n_a);
			std::vector<float> action_a = agent_a.ReturnAction(currentState_a, actionID_a);
			std::vector<int> actionID_c(actionID.begin() + n_a, actionID.end());
			std::vector<float> action_c = agent_c.ReturnAction(currentState_c, actionID_c);
			
			action_a.insert(action_a.end(), action_c.begin(), action_c.end());
			action = action_a;
			newState = env_.Step(action, reward, terminate_t);

			if (terminate_t) {
				break;
				std::cout << "Episode terminated**" << std::endl;
			}

			std::vector<float> reward_a(reward.begin(), reward.begin() + n_a);
			std::vector<float> newState_a(newState.begin(), newState.begin() + n_a);

			std::vector<float> reward_c(reward.begin() + n_a, reward.end());
			std::vector<float> newState_c(newState.begin() + n_a, newState.end());

			for (int i = 0; i < n_a; i++) {
				currentStateList_a[i].push_back(currentState_a[i]);
				actionIDList_a[i].push_back(actionID_a[i]);
				rewardList_a[i].push_back(reward_a[i]);
				newStateList_a[i].push_back(newState_a[i]);
			}

			for (int i = 0; i < n_c; i++) {
				currentStateList_c[i].push_back(currentState_c[i]);
				actionIDList_c[i].push_back(actionID_c[i]);
				rewardList_c[i].push_back(reward_c[i]);
				newStateList_c[i].push_back(newState_c[i]);
			}

			currentState = newState;

			if (currentStateList_c[0].size() >= bufferSize) {
				for (int i = 0; i < currentStateList_c.size(); i++) {
					std::vector<float> _currentState = currentStateList_c[i];
					std::vector<int> _actionID = actionIDList_c[i];
					std::vector<float> _reward = rewardList_c[i];
					std::vector<float> _newState = newStateList_c[i];
					agent_c.UpdateSVTable(_currentState, _actionID, _reward, _newState);
				}
				agent_c.SortStateValueList();
				agent_c.UpdateTPMatrix();
				for (int i = 0; i < currentStateList_c.size(); i++) {
					currentStateList_c[i].clear();
					actionIDList_c[i].clear();
					rewardList_c[i].clear();
					newStateList_c[i].clear();
				}
			}

			if (currentStateList_a[0].size() >= bufferSize) {
				for (int i = 0; i < currentStateList_a.size(); i++) {
					std::vector<float> _currentState = currentStateList_a[i];
					std::vector<int> _actionID = actionIDList_a[i];
					std::vector<float> _reward = rewardList_a[i];
					std::vector<float> _newState = newStateList_a[i];
					agent_a.UpdateSVTable(_currentState, _actionID, _reward, _newState);
				}
				agent_a.SortStateValueList();
				agent_a.UpdateTPMatrix();
				for (int i = 0; i < currentStateList_a.size(); i++) {
					currentStateList_a[i].clear();
					actionIDList_a[i].clear();
					rewardList_a[i].clear();
					newStateList_a[i].clear();
				}
			}
		}

		if (terminate_t) continue;

		float normActiveWork = env_.returnAllActiveWork();
		float activeWork_a = env_.returnActiveWork_a();
		float activeWork_c = env_.returnActiveWork_c();
		float currentNormActiveWork = env_.returnCurrentActiveWork();
		agent_c.UpdateEpsilonDecay(currentEpisode, totalEpisode);
		agent_c.UpdateLearningRateDecay(currentEpisode, totalEpisode);
		agent_a.UpdateEpsilonDecay(currentEpisode, totalEpisode);
		agent_a.UpdateLearningRateDecay(currentEpisode, totalEpisode);
		std::cout << "episode ended: " << currentEpisode << std::endl;
		std::cout << "seconds used : " << time(NULL) - currentTime << std::endl;
		std::cout << "active work is : " << normActiveWork << std::endl;
		std::cout << "current active work is : " << currentNormActiveWork << std::endl;
		std::cout << "------------------------" << std::endl;
		saveEpisodeData(simDir, currentEpisode, agent_c, agent_a);
		updateSimulationResult(simDir, "result_all", normActiveWork);
		updateSimulationResult(simDir, "result_a", activeWork_a);
		updateSimulationResult(simDir, "result_c", activeWork_c);
		updateEpsilon(simDir, agent_c.returnEpsilon());
		currentEpisode++;
		env_.t = 0;
	}
	std::cout << "Simulation completed: " << sim_id << std::endl;
	std::cout << "------------------------" << std::endl;
	currentEpisode = 1;
}

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
			agent_c.LoadDTable("DTable");
			agent_c.LoadSVTable("SVTable");
			agent_c.LoadTPMatrix("TPMatrix");
			agent_c.setEpsilon(0);
			agent_c.SortStateValueList();

			agent_a.LoadDTable("DTable_a");
			agent_a.LoadSVTable("SVTable_a");
			agent_a.LoadTPMatrix("TPMatrix_a");
			agent_a.setEpsilon(0);
			agent_a.SortStateValueList();
			resultTimer();
		}
		glutMainLoop();
	} else {
		std::vector<std::thread> threads;
		std::string dir = initSimulationData(startTime);
		for (int i = 1; i <= totalSimulation; ++i)
			threads.emplace_back(std::thread(training, dir, i));
		//wait for them to complete
		for (auto& th : threads) {
			th.join();
		}
		
	}
	std::cin.get();
	return 0;
}

void trainTimer(int)
{
	display();

	//action = agent.ReturnAction(currentState, actionID);
	//newState = env.Step(action, reward, terminate_t);

	if (terminate_t) std::cout << "invalid terminate" << std::endl;

	//agent.UpdateSVTable(currentState, actionID, reward, newState);
	//agent.UpdateEpsilonDecay(t, T*100);
	//currentState = newState;
	glutTimerFunc(1, trainTimer, 0);
}

void resultTimer(int) {
	display();

	std::vector<float> currentState_a(currentState.begin(), currentState.begin() + n_a);
	std::vector<int> actionID_a(actionID.begin(), actionID.begin() + n_a);
	std::vector<float> action_a = agent_a.ReturnAction(currentState_a, actionID_a);

	std::vector<float> currentState_c(currentState.begin() + n_a, currentState.end());
	std::vector<int> actionID_c(actionID.begin() + n_a, actionID.end());
	std::vector<float> action_c = agent_c.ReturnAction(currentState_c, actionID_c);

	action_a.insert(action_a.end(), action_c.begin(), action_c.end());

	std::vector<float> newState = env.Step(action_a, reward, terminate_t);

	if (terminate_t) std::cout << "invalid terminate" << std::endl;

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
	if (p.type == collective)
		glColor3f(255, 255, 0);
	else
		glColor3f(255, 0, 0);

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
	std::string dir = "data\\";
	std::string dirHead = "data_";
	std::string propFileName = "properties.txt";
	dir = dir + dirHead + std::to_string(time);
	int check = _mkdir(dir.c_str());

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
	outFile << "learning rate = " << agent_c.learningRate << '\n';
	outFile << "learning rate decay = " << agent_c.learningRateDecay << '\n';
	outFile << "epsilon = " << agent_c.epsilon << '\n';
	outFile << "epsilon decay = " << agent_c.epsilonDecay << '\n';
	outFile << "discount factor = " << agent_c.discountFactor << '\n';
	outFile << "noise = " << '\n';
	outFile << "range = " << '\n';
	outFile << "density = " << '\n';
	outFile << "reward func = " << '\n';
	outFile << "learner count = " << n << '\n';
	outFile << "remark = ''" << '\n';
	outFile.close();

	return dir;
}

std::string createSimulationDir(std::string dir, int sim) {
	std::string subDir = "sim_";
	dir = dir + "\\" + subDir + std::to_string(sim);
	int check = _mkdir(dir.c_str());

	if (!check)
		std::cout << "Directory created: " << dir << std::endl;
	else {
		std::cout << "Failed to created directory: " << dir << std::endl;
		exit(1);
	}
	return dir;
}

void saveEpisodeData(std::string dir, int episode, Agent ag_c, Agent ag_a) {
	std::string subDir = "ep_";
	dir = dir + "\\" + subDir + std::to_string(episode);
	int check = _mkdir(dir.c_str());

	if (!check)
		std::cout << "Directory created: " << dir << std::endl;
	else {
		std::cout << "Failed to created directory: " << dir << std::endl;
		exit(1);
	}

	ag_c.SaveQTable((dir + "\\" + "QTable_c").c_str());
	ag_c.SaveSVTable((dir + "\\" + "SVTable_c").c_str());
	ag_c.SaveDTable((dir + "\\" + "DTable_c").c_str());
	ag_c.SaveTPMatrix((dir + "\\" + "TPMatrix_c").c_str());

	ag_a.SaveQTable((dir + "\\" + "QTable_a").c_str());
	ag_a.SaveSVTable((dir + "\\" + "SVTable_a").c_str());
	ag_a.SaveDTable((dir + "\\" + "DTable_a").c_str());
	ag_a.SaveTPMatrix((dir + "\\" + "TPMatrix_a").c_str());
}

void updateSimulationResult(std::string dir, std::string name, float activeWork) {
	dir = dir + "\\" + name + ".csv";
	std::ofstream outFile(dir, std::ofstream::app);
	outFile << activeWork << '\n';
	outFile.close();
}

void updateEpsilon(std::string dir, float epsilon) {
	std::string resultFileName = "epsilon.csv";
	dir = dir + "\\" + resultFileName;
	std::ofstream outFile(dir, std::ofstream::app);
	outFile << epsilon << '\n';
	outFile.close();
}

void readConfig() {

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
			//std::cout << "Print Q Table : " << t << std::endl;
			for (int i = 0; i < 21; ++i)
			{
				for (int j = 0; j < 21; ++j)
				{
					std::cout << agent_c.QTable[i][j] << '\t';
				}
				std::cout << std::endl << "----------------------------" << std::endl;
			}
			agent_c.SaveQTable("QTable");
			agent_c.SaveSVTable("SVTable");
			std::cin.get();
			break;
		}
		case 50: {
			// "2"
			std::cout << "Print agent status:" << std::endl;
			std::cout << "----------------------------" << std::endl;
			std::cout << "current epsilon: " << agent_c.returnEpsilon() << std::endl;
			std::cout << "total active work : " << env.returnAllActiveWork() << std::endl;
			std::cout << "current active work : " << env.returnCurrentActiveWork() << std::endl;
			std::cout << "seconds passed: " << std::to_string(time(NULL) - startTime) << std::endl;
			std::cout << "----------------------------" << std::endl;
			break;
		}
		case 51: {
			// "3"
			//std::cout << "Print time : " << t << std::endl;
			std::cin.get();
			break;
		}
		case 52: {
			// "4"
			std::cout << "Loading Q Table from CSV..." << std::endl;
			agent_c.LoadQTable("QTable");
			break;
		}
	}
}