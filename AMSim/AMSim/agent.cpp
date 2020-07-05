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

void Agent::UpdateQTable(float state, int actionID, float reward, float newState) {
	int stateID = StateToIndex(state);
	int newStateID = StateToIndex(newState);

	//float lr = learningRate * (1 - ((float)DSATable[stateID][actionID] / (float)N));
	//QTable[stateID][actionID] += lr * (reward + discountFactor * arrmax(QTable[newStateID], sizeof(QTable[newStateID])) - QTable[stateID][actionID]);
	//QTable[stateID][actionID] += lr * (reward - QTable[stateID][actionID]);
	SVTable[stateID] += learningRate * (reward - SVTable[stateID]);
	DTable[stateID][newStateID][actionID]++;
	//DSATable[stateID][actionID]++;
	//N++;
}

void Agent::SortStateValueList() {
	std::vector<int> V(pieces);
	std::iota(V.begin(), V.end(), 0);
	sort(V.begin(), V.end(), [&](int i, int j) {return SVTable[i]>SVTable[j]; });
	for (int k = 0; k < V.size(); k++)
		sortedSVTable[k] = V[k];
}

void Agent::UpdateQTable(std::vector<float> stateList, std::vector<int> actionIDList, std::vector<float> rewardList, std::vector<float> newStateList) {
	for (int i = 0; i < stateList.size(); i++) {
		UpdateQTable(stateList[i], actionIDList[i], rewardList[i], newStateList[i]);
	}
}

void Agent::UpdateTPMatrix() {
	int n_d1 = sizeof(TPMatrix) / sizeof(*TPMatrix);
	int n_d2 = sizeof(TPMatrix[n_d1]) / sizeof(*TPMatrix[n_d1]);
	int n_d3 = sizeof(TPMatrix[n_d1][n_d2]) / sizeof(*TPMatrix[n_d1][n_d2]);
	for (int i = 0; i < n_d3; i++) {
		for (int j = 0; j < n_d1; j++) {
			int actionTotal = 0;
			for (int k = 0; k < n_d2; k++) {
				actionTotal += DTable[j][k][i];
			}
			for (int k = 0; k < n_d2; k++) {
				TPMatrix[j][k][i] = (float)DTable[j][k][i] / (float)actionTotal;
			}
		}
	}
}

float Agent::ReturnAction(float state, int &actionID) {
	int stateID = StateToIndex(state);
	int count = sizeof(QTable[stateID]) / sizeof(*QTable[stateID]);
	if (epsilon >= ((double)rand() / (RAND_MAX))) {
		actionID = rand() % count;
	} else {
		//TP-matrix
		//int optimumStateID = argmax(SVTable, count);
		//actionID = argmax(TPMatrix[stateID][optimumStateID], count);
		float thr = 0.1;
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

		// Q-learning
		//actionID = argmax(QTable[stateID], count);
	}
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

int argmax(float *arr, int size) {
	return std::distance(arr, std::max_element(arr ,arr + size));
}

float arrmax(float *arr, int size) {
	float max = arr[0];
	for(int i = 1; i < size; i++)
		if (max < arr[i]) max = arr[i];
	return max;
}

int Agent::StateToIndex(float state) {
	state += M_PI;
	return floor(state / radiansPiece);
}

int Agent::ActionToIndex(float state) {
	state += M_PI;
	return floor(state / radiansPiece);
}

float Agent::IndexToAction(int index) {
	return (radiansPiece * index) - M_PI;
}

void Agent::SaveQTable(const char* path) {
	std::string fileExt = ".csv";
	std::string filePath = path + fileExt;
	std::ofstream outFile(filePath);
	for (auto& row : QTable) {	
		for (auto col : row)
			outFile << col << ',';
		outFile << '\n';
	}
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

void Agent::LoadQTable(const char* path) {
	std::string fileExt = ".csv";
	std::string filePath = path + fileExt;
	std::cout << "loading Q table from " << filePath << std::endl;
	std::ifstream file(filePath);
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

void Agent::LoadSVTable(const char* path) {
	std::string fileExt = ".csv";
	std::string filePath = path + fileExt;
	std::cout << "loading SV Table from svtable.csv" << std::endl;
	std::ifstream file(filePath);
	for (int row = 0; row < pieces; row++)
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
