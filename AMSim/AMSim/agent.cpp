#define _USE_MATH_DEFINES

#include "agent.h"
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm> 
#include <fstream>
#include <sstream>
#include <functional>
#include <numeric>

int argmax(float*, int);
float arrmax(float*, int);

Agent::Agent(float lr, float ep, float epDecay, float epMin, float df)
{
	learningRate = lr;
	maxlr = learningRate;
	epsilon = ep;
	epsilonDecay = epDecay;
	minEpsilon = epMin;
	discountFactor = df;
}

// sort optimum state
void Agent::SortStateValueList() {
	std::vector<int> V(K);
	std::iota(V.begin(), V.end(), 0);
	sort(V.begin(), V.end(), [&](int i, int j) {return SVTable[i]>SVTable[j]; });
	for (int k = 0; k < V.size(); k++)
		sortedSVTable[k] = V[k];
}

// state value function update via TD-Learning
void Agent::UpdateSVTable(float state, int actionID, float reward, float newState) {
	int stateID = StateToIndex(state);
	int newStateID = StateToIndex(newState);
	SVTable[stateID] += learningRate * (reward + discountFactor * SVTable[newStateID] - SVTable[stateID]); // TD-Learning
	DTable[stateID][newStateID][actionID]++; // record state-action occurance in a distribution table
}

// update state value for MAS
void Agent::UpdateSVTable(std::vector<float> stateList, std::vector<int> actionIDList, std::vector<float> rewardList, std::vector<float> newStateList) {
	for (int i = 0; i < stateList.size(); i++) {
		UpdateSVTable(stateList[i], actionIDList[i], rewardList[i], newStateList[i]);
	}
}

// model update
void Agent::UpdateTPMatrix() {
	int n_d1 = sizeof(TPMatrix) / sizeof(*TPMatrix);
	int n_d2 = sizeof(TPMatrix[n_d1]) / sizeof(*TPMatrix[n_d1]);
	int n_d3 = sizeof(TPMatrix[n_d1][n_d2]) / sizeof(*TPMatrix[n_d1][n_d2]);
	for (int i = 0; i < n_d3; i++) {
		for (int j = 0; j < n_d1; j++) {
			int actionTotal = 0;
			// compute total occurance of action row
			for (int k = 0; k < n_d2; k++) {
				actionTotal += DTable[j][k][i]; 
			}
			// compute probability of state-action to next state
			for (int k = 0; k < n_d2; k++) {
				TPMatrix[j][k][i] = (float)DTable[j][k][i] / (float)actionTotal;
			}
		}
	}
}

// action selection function
float Agent::ReturnAction(float state, int &actionID) {
	int stateID = StateToIndex(state);
	int count = sizeof(SVTable) / sizeof(SVTable[stateID]);
	if (epsilon >= ((double)rand() / (RAND_MAX))) {
		actionID = rand() % count;
	} else {
		/*TP-Matrix (model) action select. one can search for local available actions with highest value,
		or just sort all state value accrodingly and look for any action agents can take (our approach).
		our approach required less computational operation needed, even it is not intuitive.*/
		float thr = 0.01;
		int index = 0;
		int SVTableCount = sizeof(sortedSVTable) / sizeof(sortedSVTable[0]);
		while (index < SVTableCount) {
			int optimumStateID = sortedSVTable[index];
			actionID = argmax(TPMatrix[stateID][optimumStateID], count);
			if (TPMatrix[stateID][optimumStateID][actionID] > thr) {
				return -(IndexToAction(actionID));
			}
			else {
				index++;
			}
		}
	}
	return -(IndexToAction(actionID)); // append negative sign to flip orientation
}

// action selection for MAS
std::vector<float> Agent::ReturnAction(std::vector<float> stateList, std::vector<int> &actionIDList) {
	std::vector<float> actionList;
	for (int i = 0; i < stateList.size(); i++) {
		actionList.push_back(ReturnAction(stateList[i], actionIDList[i]));
	}
	return actionList;
}

void Agent::UpdateEpsilonDecay(float t, float totalTime) {
	epsilon = minEpsilon + (maxEpsilon - minEpsilon) * exp(-epsilonDecay * t / totalTime);
}

void Agent::UpdateLearningRateDecay(float t, float totalTime) {
	learningRate = minlr + (maxlr - minlr) * exp(-learningRateDecay * t / totalTime);
}

void Agent::setEpsilon(float ep) {
	epsilon = ep;
}

float Agent::returnEpsilon() {
	return epsilon;
}

void Agent::setLearningRate(float lr) {
	learningRate = lr;
}

float Agent::returnLearningRate() {
	return learningRate;
}

// return max value index within array
int argmax(float *arr, int size) {
	return std::distance(arr, std::max_element(arr ,arr + size));
}

// return max value within array
float arrmax(float *arr, int size) {
	float max = arr[0];
	for(int i = 1; i < size; i++)
		if (max < arr[i]) max = arr[i];
	return max;
}

// from state in [-pi, pi) to state index in [0, K]
int Agent::StateToIndex(float state) {
	/*
	this help to put external force into the middle of a state piece
	doing so help to avoid the behaviour where flocking agents will going around in a circle
	it is not implemented because previous training was not done using it
	uncomment this to make the simulation look more natural 
	*/
	//state += M_PI + (radiansPiece / 2); 
	
	state += M_PI; 
	return floor(state / radiansPiece);
}

// from action in [-pi, pi) to action index in [0, K]
int Agent::ActionToIndex(float state) {
	state += M_PI;
	return floor(state / radiansPiece);
}

// from action index in [0, K] to action in [-pi, pi)
float Agent::IndexToAction(int index) {
	return (radiansPiece * index) - M_PI;
}

void Agent::SaveSVTable(const char* path) {
	std::string fileExt = ".csv";
	std::string filePath = path + fileExt;
	std::ofstream outFile(filePath);
	for (auto& row : SVTable) {
			outFile << row << '\n';
	}
}

void Agent::SaveTPMatrix(const char* path) {
	std::string fileExt = ".csv";
	std::string filePath = path + fileExt;
	FILE* pFile = fopen(path, "wb");
	fwrite(TPMatrix, sizeof(TPMatrix), 1, pFile);
	fclose(pFile);

	std::ofstream outFile(filePath);
	int n_d1 = sizeof(TPMatrix) / sizeof(*TPMatrix);
	int n_d2 = sizeof(TPMatrix[n_d1]) / sizeof(*TPMatrix[n_d1]);
	int n_d3 = sizeof(TPMatrix[n_d1][n_d2]) / sizeof(*TPMatrix[n_d1][n_d2]);
	for (int i = 0; i < n_d3; i++) {
		for (int j = 0; j < n_d1; j++) {
			for (int k = 0; k < n_d2; k++) {
				outFile << TPMatrix[j][k][i] << ',';
			}
			outFile << '\n';
		}
		outFile << '\n';
		outFile << '\n';
	}
}

void Agent::SaveDTable(const char* path) {
	std::string fileExt = ".csv";
	std::string filePath = path + fileExt;
	FILE* pFile = fopen(path, "wb");
	fwrite(DTable, sizeof(DTable), 1, pFile);
	fclose(pFile);

	std::ofstream outFile(filePath);
	int n_d1 = sizeof(DTable) / sizeof(*DTable);
	int n_d2 = sizeof(DTable[n_d1]) / sizeof(*DTable[n_d1]);
	int n_d3 = sizeof(DTable[n_d1][n_d2]) / sizeof(*DTable[n_d1][n_d2]);
	for (int i = 0; i < n_d3; i++) {
		for (int j = 0; j < n_d1; j++) {
			for (int k = 0; k < n_d2; k++) {
				outFile << DTable[j][k][i] << ',';
			}
			outFile << '\n';
		}
		outFile << '\n';
		outFile << '\n';
	}
}

void Agent::LoadSVTable(const char* path) {
	std::string fileExt = ".csv";
	std::string filePath = path + fileExt;
	std::cout << "loading SV Table from svtable.csv" << std::endl;
	std::ifstream file(filePath);
	for (int row = 0; row < K; row++)
	{
		std::string line;
		std::getline(file, line);
		std::stringstream iss(line);	
		iss >> SVTable[row];
	}
}

void Agent::LoadTPMatrix(const char* path) {
	FILE* pFile = fopen(path, "rb");
	fread(TPMatrix, sizeof(TPMatrix), 1, pFile);
	fclose(pFile);
}

void Agent::LoadDTable(const char* path) {
	FILE* pFile = fopen(path, "rb");
	fread(DTable, sizeof(DTable), 1, pFile);
	fclose(pFile);
}

Agent::~Agent()
{
}
