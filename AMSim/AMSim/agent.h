#ifndef AGENT_H
#define AGENT_H

// redefine M_PI to fix imcompatability with CUDA
#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#include <string>
#include <vector>
#include <cmath>

static const int K = 20;  // split state space and action space into K pieces of discretized units

class Agent {
public:
	Agent(float = 0.5, float = 1, float = 10, float = 0 , float = 0.9);
	const float RADIANS = M_PI * 2;
	float learningRate = 0.1; // learning rate alpha
	float minlr = 0; // minimum learning rate
	float maxlr = learningRate; // maximum learning rate
	float learningRateDecay = 20; // learning rate decay lambda
	float discountFactor = 0.2; // discount factor gamma
	float epsilon = 1; // exploration rate epsilon
	float minEpsilon = 0; // minimum exploration rate
	float maxEpsilon = epsilon; // maximum exploration rate
	float epsilonDecay = 10; // exploration rate decay lambda
	float radiansPiece = RADIANS / (float)K; // descretized units of state and action in radians
	float SVTable[K] = {0}; // state value table
	int sortedSVTable[K] = {0}; // sorted state value table, required during action selection with model
	float TPMatrix[K][K][K] = {{{0}}}; // model, storing transition probability of state-action to next state
	int DTable[K][K][K] = {{{0}}}; // distribution table store occurance of state-action

	void UpdateSVTable(float, int, float, float);
	void UpdateSVTable(std::vector<float>, std::vector<int>, std::vector<float>, std::vector<float>);
	void SortStateValueList();
	void UpdateTPMatrix();
	float ReturnAction(float, int&);
	std::vector<float> ReturnAction(std::vector<float>, std::vector<int>&);
	void UpdateEpsilonDecay(float, float);
	void setEpsilon(float);
	float returnEpsilon();
	void UpdateLearningRateDecay(float, float);
	void setLearningRate(float);
	float returnLearningRate();
	void SaveSVTable(const char*);
	void SaveTPMatrix(const char*);
	void SaveDTable(const char*);
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