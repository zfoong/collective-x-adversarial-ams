#define _USE_MATH_DEFINES

#include "agent.h"
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm> 

const float RADIANS = M_PI * 2;
float learningRate = 0.1;
float discountFactor = 0.2;
float epsilon = 1.0;
float epsilonDecay = 0.1;

int argmax(float*, int);
float arrmax(float*, int);
int StateToIndex(float);
int ActionToIndex(float);

Agent::Agent()
{

}

void Agent::UpdateQTable(float state, int actionID, float reward, float newState) {
	int stateID = StateToIndex(state);
	int newStateID = StateToIndex(newState);

	QTable[stateID][actionID] += learningRate * (reward + discountFactor * arrmax(QTable[newStateID], sizeof(QTable[newStateID])) - QTable[stateID][actionID]);
}

void Agent::UpdateQTable(std::vector<float> stateList, std::vector<int> actionIDList, std::vector<float> rewardList, std::vector<float> newStateList) {
	for (int i = 0; i < stateList.size(); i++) {
		UpdateQTable(stateList[i], actionIDList[i], rewardList[i], newStateList[i]);
	}
}

float Agent::ReturnAction(float state, int &actionID) {
	int stateID = StateToIndex(state);
	int count = sizeof(QTable[stateID]) / sizeof(*QTable[stateID]);
	if (epsilon > ((double)rand() / (RAND_MAX))) {
		actionID = rand() % count;
	}
	else {
		actionID = argmax(QTable[stateID], count);
	}
	return QTable[stateID][actionID];
}

std::vector<float> Agent::ReturnAction(std::vector<float> stateList, std::vector<int> &actionIDList) {
	std::vector<float> actionList;
	for (int i = 0; i < stateList.size(); i++) {
		actionList.push_back(ReturnAction(stateList[i], actionIDList[i]));
	}
	return actionList;
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

int StateToIndex(float radians) {
	float radiansPiece =  RADIANS / 16;
	// TODO: floor vs round
	int index = floor(radians / radiansPiece);
	return index;
}

int ActionToIndex(float radians) {
	float radiansPiece = RADIANS / 16;
	int index = floor(radians / radiansPiece);
	return index;
}

void Agent::SaveQTable() {

}

void Agent::LoadQTable() {

}

Agent::~Agent()
{

}
