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
#include <thread>

void resultTimer(int = 0);
void display();
void keyboard(unsigned char, int, int);
void drawMatter(Matter, float = 0, float = 0);
std::string initSimulationData(int);
std::string createSimulationDir(std::string, int);
void saveEpisodeData(std::string, int, Agent, Agent);
void updateSimulationResult(std::string, std::string, float);
void updateEpsilon(std::string, float);

extern float windowWidth;
extern float windowHeight;
extern const float SCALE_FACTOR;
extern float scaledWindowWidth;
extern float scaledWindowHeight;
extern float dt;

int totalSimulation = 10; // amt of simulation run in parallel
int currentEpisode_1 = 1;
int totalEpisode = 120; // total episode
float T = 100; // total Timestep
bool terminate_t = false;
bool isLearning = true;

int WIN;
int startTime = time(NULL);

int N_f = 2; // amt of flocking agents
int N_c = 30; // amt of clustering agents
Environment env = Environment(N_f, N_c, true); // init environment
Agent agent_f = Agent(0.1, 0.2, 10, 0, 0.8); // init flocking agents
Agent agent_c = Agent(0.1, 0.2, 10, 0, 0.8); // init clustering agents

std::vector<float> currentState = env.ReturnAllState(); // get initial state from environment
int n = N_c + N_f;
std::vector<int> actionID(n);
std::vector<float> reward(n);

int bufferSize = 10000; // buffer size threshold beta, when exceed, agent perform learning

void training(std::string dir, int sim_id) {
	std::string simDir = createSimulationDir(dir, sim_id); // create file path that store simulation data
	Agent agent_f = Agent(0.1, 0.2, 10, 0, 0.8); // init flocking agent
	Agent agent_c = Agent(0.1, 0.2, 10, 0, 0.8); // init clustering agent

	// transfer learned model and value function for flocking agent
	agent_f.LoadDTable("DTable_f");
	agent_f.LoadSVTable("SVTable_f");
	agent_f.LoadTPMatrix("TPMatrix_f");
	agent_f.SortStateValueList();

	// transfer learned model and value function for clustering agent
	agent_c.LoadDTable("DTable_c");
	agent_c.LoadSVTable("SVTable_c");
	agent_c.LoadTPMatrix("TPMatrix_c");
	agent_c.SortStateValueList();

	std::cout << "------------------------" << std::endl;
	std::cout << "Simulation started: " << sim_id << std::endl;
	int currentEpisode = 1;

	while (currentEpisode <= totalEpisode) {
		Environment env_ = Environment(N_f, N_c); // init environment
		std::vector<float> currentState = env_.ReturnAllState(); // return initial state

		// init joint action, state, reward set
		std::vector<int> actionID(n);
		std::vector<float> reward(n);
		std::vector<float> newState(n);
		std::vector<float> action(n);

		// experience buffer that store joint action, current state, next state and reward set for clustering agent
		std::vector<std::vector<float>> currentStateList_c(N_c);
		std::vector<std::vector<int>> actionIDList_c(N_c);
		std::vector<std::vector<float>> rewardList_c(N_c);
		std::vector<std::vector<float>> newStateList_c(N_c);

		// experience buffer that store joint action, current state, next state and reward set for flocking agent
		std::vector<std::vector<float>> currentStateList_f(N_f);
		std::vector<std::vector<int>> actionIDList_f(N_f);
		std::vector<std::vector<float>> rewardList_f(N_f);
		std::vector<std::vector<float>> newStateList_f(N_f);

		int currentTime = time(NULL);
		terminate_t = false;
		std::cout << "current epsilon is : " << agent_f.returnEpsilon() << std::endl;
		std::cout << "current learning rate is : " << agent_f.returnLearningRate() << std::endl;
		std::cout << "episode started: " << currentEpisode << std::endl;

		while (env_.t < T)
		{
			// split joint state (action) into flocking and clustering agent joint state (action)
			std::vector<float> currentState_c(currentState.begin(), currentState.begin() + N_c);
			std::vector<float> currentState_f(currentState.begin() + N_c, currentState.end());
			std::vector<int> actionID_c(actionID.begin(), actionID.begin() + N_c);
			std::vector<float> action_c = agent_c.ReturnAction(currentState_c, actionID_c);
			std::vector<int> actionID_f(actionID.begin() + N_c, actionID.end());
			std::vector<float> action_f = agent_f.ReturnAction(currentState_f, actionID_f);
			
			// joint action set
			action_c.insert(action_c.end(), action_f.begin(), action_f.end());
			action = action_c;

			// return action set to environment and receive reward and next state set
			newState = env_.Step(action, reward, terminate_t);

			// terminate only happen when environment detect abnormal behaviour from particles (such as repel or stacking)
			if (terminate_t) {
				std::cout << "Episode terminated** Invalid particles movement, please decrease time step dt" << std::endl;
				break;
			}

			// split reward set into flocking and clustering agent reward set
			std::vector<float> reward_a(reward.begin(), reward.begin() + N_c);
			std::vector<float> newState_a(newState.begin(), newState.begin() + N_c);
			std::vector<float> reward_c(reward.begin() + N_c, reward.end());
			std::vector<float> newState_c(newState.begin() + N_c, newState.end());

			// store experience in clustering agent experience buffer
			for (int i = 0; i < N_c; i++) {
				currentStateList_c[i].push_back(currentState_c[i]);
				actionIDList_c[i].push_back(actionID_c[i]);
				rewardList_c[i].push_back(reward_a[i]);
				newStateList_c[i].push_back(newState_a[i]);
			}

			// store experience in flocking agent experience buffer
			for (int i = 0; i < N_f; i++) {
				currentStateList_f[i].push_back(currentState_f[i]);
				actionIDList_f[i].push_back(actionID_f[i]);
				rewardList_f[i].push_back(reward_c[i]);
				newStateList_f[i].push_back(newState_c[i]);
			}

			// reset new state to current state
			currentState = newState;

			// when experience buffer exceed beta, update flocking agents model and value function (learning), and empty experience buffer
			if (currentStateList_f[0].size() >= bufferSize) {
				for (int i = 0; i < currentStateList_f.size(); i++) {
					std::vector<float> _currentState = currentStateList_f[i];
					std::vector<int> _actionID = actionIDList_f[i];
					std::vector<float> _reward = rewardList_f[i];
					std::vector<float> _newState = newStateList_f[i];
					agent_f.UpdateSVTable(_currentState, _actionID, _reward, _newState);
				}
				agent_f.SortStateValueList();
				agent_f.UpdateTPMatrix();
				for (int i = 0; i < currentStateList_f.size(); i++) {
					currentStateList_f[i].clear();
					actionIDList_f[i].clear();
					rewardList_f[i].clear();
					newStateList_f[i].clear();
				}
			}

			// when experience buffer exceed beta, update clustering agents model and value function (learning), and empty experience buffer
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
		}

		if (terminate_t) continue;

		float normActiveWork = env_.returnAllActiveWork();
		float activeWork_c = env_.returnActiveWork_c();
		float activeWork_f = env_.returnActiveWork_f();
		agent_f.UpdateEpsilonDecay(currentEpisode, totalEpisode);
		agent_f.UpdateLearningRateDecay(currentEpisode, totalEpisode);
		agent_c.UpdateEpsilonDecay(currentEpisode, totalEpisode);
		agent_c.UpdateLearningRateDecay(currentEpisode, totalEpisode);
		std::cout << "episode ended: " << currentEpisode << std::endl;
		std::cout << "seconds used : " << time(NULL) - currentTime << std::endl;
		std::cout << "active work for all agent is : " << normActiveWork << std::endl;
		std::cout << "active work for clustering agent is : " << activeWork_c << std::endl;
		std::cout << "active work for flocking agent is : " << activeWork_f << std::endl;
		std::cout << "------------------------" << std::endl;
		saveEpisodeData(simDir, currentEpisode, agent_f, agent_c);
		updateSimulationResult(simDir, "result_all", normActiveWork);
		updateSimulationResult(simDir, "result_c", activeWork_c);
		updateSimulationResult(simDir, "result_f", activeWork_f);
		updateEpsilon(simDir, agent_f.returnEpsilon());
		currentEpisode++;
		env_.t = 0;
	}
	std::cout << "Simulation completed: " << sim_id << std::endl;
	std::cout << "------------------------" << std::endl;
	currentEpisode = 1;
}

int main(int argc, char **argv)
{
	// create '/data' directory
	int check = _mkdir("data");
	if (!check)
		std::cout << "Directory created: " << "data" << std::endl;


	if (!isLearning) {
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

		agent_f.LoadDTable("DTable_f");
		agent_f.LoadSVTable("SVTable_f");
		agent_f.LoadTPMatrix("TPMatrix_f");
		agent_f.setEpsilon(0);
		agent_f.SortStateValueList();

		agent_c.LoadDTable("DTable_c");
		agent_c.LoadSVTable("SVTable_c");
		agent_c.LoadTPMatrix("TPMatrix_c");
		agent_c.setEpsilon(0);
		agent_c.SortStateValueList();
		resultTimer();
		
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

void resultTimer(int) {
	display();

	std::vector<float> currentState_c(currentState.begin(), currentState.begin() + N_c);
	std::vector<int> actionID_c(actionID.begin(), actionID.begin() + N_c);
	std::vector<float> action_c = agent_c.ReturnAction(currentState_c, actionID_c);

	std::vector<float> currentState_f(currentState.begin() + N_c, currentState.end());
	std::vector<int> actionID_f(actionID.begin() + N_c, actionID.end());
	std::vector<float> action_f = agent_f.ReturnAction(currentState_f, actionID_f);

	action_c.insert(action_c.end(), action_f.begin(), action_f.end());

	std::vector<float> newState = env.Step(action_c, reward, terminate_t);

	if (terminate_t) std::cout << "invalid terminate" << std::endl;

	currentState = newState;

	glutTimerFunc(1, resultTimer, 0);
}

// draw on screen with GLUT
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

// draw matter on screen
void drawMatter(Matter p, float transformX, float transformY) {
	glPushMatrix();
	glScalef(SCALE_FACTOR, SCALE_FACTOR, 1.0);
	glTranslatef(p.pos[0] + transformX, p.pos[1] + transformY, 0.0f);
	if (p.type == flocking)
		glColor3f(255, 255, 0); // yellow for flocking agent
	else
		glColor3f(255, 0, 0); // red for clustering agent

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
	outFile << "clustering agent count = " << N_c << '\n';
	outFile << "flocking agent count = " << N_f << '\n';
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

void saveEpisodeData(std::string dir, int episode, Agent ag_f, Agent ag_c) {
	std::string subDir = "ep_";
	dir = dir + "\\" + subDir + std::to_string(episode);
	int check = _mkdir(dir.c_str());

	if (!check)
		std::cout << "Directory created: " << dir << std::endl;
	else {
		std::cout << "Failed to created directory: " << dir << std::endl;
		exit(1);
	}

	ag_f.SaveSVTable((dir + "\\" + "SVTable_f").c_str());
	ag_f.SaveDTable((dir + "\\" + "DTable_f").c_str());
	ag_f.SaveTPMatrix((dir + "\\" + "TPMatrix_f").c_str());

	ag_c.SaveSVTable((dir + "\\" + "SVTable_c").c_str());
	ag_c.SaveDTable((dir + "\\" + "DTable_c").c_str());
	ag_c.SaveTPMatrix((dir + "\\" + "TPMatrix_c").c_str());
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

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 27: {
			// keyboard key "esc"
			glutDestroyWindow(WIN);
			exit(0);
		}
		case 49: {
			// keyboard key "1"
			std::cout << "Print agents status:" << std::endl;
			std::cout << "----------------------------" << std::endl;
			std::cout << "current epsilon: " << agent_c.returnEpsilon() << std::endl;
			std::cout << "total active work : " << env.returnAllActiveWork() << std::endl;
			std::cout << "seconds passed: " << std::to_string(time(NULL) - startTime) << std::endl;
			std::cout << "----------------------------" << std::endl;
			break;
		}
	}
}