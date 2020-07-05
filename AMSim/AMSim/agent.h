#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <vector>

const int pieces = 20;

class Agent {
public:
	const float RADIANS = M_PI * 2;
	float learningRate = 0.1;
	float minlr = 0;
	float maxlr = learningRate;
	float learningRateDecay = 20;
	float discountFactor = 0.2;
	float epsilon = 1;
	float minEpsilon = 0;
	float maxEpsilon = epsilon;
	float epsilonDecay = 10;
	float radiansPiece = RADIANS / (float)pieces;
	Agent(float = 0.5, float = 1, float = 10, float = 0 , float = 0.9);
	float QTable[pieces][pieces] = {{0}};
	float SVTable[pieces] = {0};
	int sortedSVTable[pieces] = { 0 };
	float TPMatrix[pieces][pieces][pieces] = {{{0}}};
	int DTable[pieces][pieces][pieces] = {{{0}}};
	int DSATable[pieces][pieces] = { 1 };
	int N = pieces*pieces;
	void UpdateQTable(float, int, float, float);
	void UpdateQTable(std::vector<float>, std::vector<int>, std::vector<float>, std::vector<float>);
	void SortStateValueList();
	void UpdateTPMatrix();
	float ReturnAction(float, int&);
	std::vector<float> ReturnAction(std::vector<float>, std::vector<int>&);
	void setEpsilon(float);
	float returnEpsilon();
	void UpdateEpsilonDecay(float, float);
	void UpdateLearningRateDecay(float, float);
	void setLearningRate(float);
	float returnLearningRate();
	void SaveQTable(const char*);
	void SaveSVTable(const char*);
	void SaveTPMatrix(const char*);
	void SaveDTable(const char*);
	void LoadQTable(const char*);
	void LoadSVTable(const char*);
	void LoadTPMatrix(const char*);
	void LoadDTable(const char*);
	virtual ~Agent();
protected:
private:
	int StateToIndex(float);
	int ActionToIndex(float);
	float IndexToAction(int);
	//Agent(const Agent& other) {}
	//Agent& operator=(const Agent& other) {}
};

#endif // AGENT_H