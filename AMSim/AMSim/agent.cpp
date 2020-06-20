#define _USE_MATH_DEFINES

#include "agent.h"
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm> 
#include <fstream>
#include <sstream>

const float RADIANS = M_PI * 2;
float learningRate = 0.1;
float discountFactor = 0.2;
float minEpsilon = 0;
float maxEpsilon = 1;
float epsilon = 1;
float epsilonDecay = 0.1;
float radiansPiece = RADIANS / (float)pieces;

int argmax(float*, int);
float arrmax(float*, int);
int StateToIndex(float);
int ActionToIndex(float);
float IndexToAction(int);

Agent::Agent(float lr, float ep, float epDecay, float epMin, float df)
{
	learningRate = lr;
	epsilon = ep;
	epsilonDecay = epDecay;
	minEpsilon = epMin;
	discountFactor = df;
}

void Agent::UpdateQTable(float state, int actionID, float reward, float newState) {
	int stateID = StateToIndex(state);
	int newStateID = StateToIndex(newState);
	QTable[stateID][actionID] += learningRate * (reward - QTable[stateID][actionID]);
	// QTable[stateID][actionID] += learningRate * (reward + discountFactor * arrmax(QTable[newStateID], sizeof(QTable[newStateID])) - QTable[stateID][actionID]);
	// QTable[stateID][actionID] += learningRate * (reward + discountFactor * QTable[newStateID][newActionID] - QTable[stateID][actionID]);
}

void Agent::UpdateQTable(std::vector<float> stateList, std::vector<int> actionIDList, std::vector<float> rewardList, std::vector<float> newStateList) {
	for (int i = 0; i < stateList.size(); i++) {
		//UpdateQTable(stateList[i], actionIDList[i], rewardList[i], newStateList[i]);
	}
}

float Agent::ReturnAction(float state, int &actionID) {
	int stateID = StateToIndex(state);
	int count = sizeof(QTable[stateID]) / sizeof(*QTable[stateID]);
	if (epsilon >= ((double)rand() / (RAND_MAX))) 
		actionID = rand() % count;
	else
		actionID = argmax(QTable[stateID], count);
	return -(IndexToAction(actionID)); // append negative sign to flip orientation
}

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

void Agent::setEpsilon(float ep) {
	epsilon = ep;
}

float Agent::returnEpsilon() {
	return epsilon;
}

int argmax(float *arr, int size) {
	return std::distance(arr, std::max_element(arr ,arr + size));
}

float arrmax(float *arr, int size) {
	float max = arr[0];
	for(int i = 1; i < size; i++)
		if (max < arr[i]) max = arr[i];
	return max;
}

int StateToIndex(float state) {
	// TODO: floor vs round
	state += M_PI;
	return floor(state / radiansPiece);
}

int ActionToIndex(float state) {
	state += M_PI;
	return floor(state / radiansPiece);
}

float IndexToAction(int index) {
	return (radiansPiece * index) - M_PI;
}

void Agent::SaveQTable() {
	std::ofstream outFile("test.csv");
	for (auto& row : QTable) {	
		for (auto col : row)
			outFile << col << ',';
		outFile << '\n';
	}
}

void Agent::LoadQTable() {
	std::cout << "loading Q table from test.csv" << std::endl;
	std::ifstream file("test.csv");
	for (int row = 0; row < pieces; row++)
	{
		std::string line;
		std::getline(file, line);
		std::stringstream iss(line);
		for (int col = 0; col < pieces; col++)
		{
			std::string val;
			std::getline(iss, val, ',');
			std::stringstream convertor(val);
			convertor >> QTable[row][col];
		}
	}
}

Agent::~Agent()
{

}
