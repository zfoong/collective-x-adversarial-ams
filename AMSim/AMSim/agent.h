#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <vector>

const int pieces = 21;

class Agent {
public:
	Agent(float = 0.01, float = 1, float = 10, float = 0 , float = 0.9);
	float QTable[pieces][pieces] = {{0}};
	void UpdateQTable(float, int, float, float);
	void UpdateQTable(std::vector<float>, std::vector<int>, std::vector<float>, std::vector<float>);
	float ReturnAction(float, int&);
	void setEpsilon(float);
	float Agent::returnEpsilon();
	std::vector<float> ReturnAction(std::vector<float>, std::vector<int>&);
	void Agent::UpdateEpsilonDecay(float, float);
	void SaveQTable();
	void LoadQTable();
	virtual ~Agent();
protected:
private:
	//Agent(const Agent& other) {}
	//Agent& operator=(const Agent& other) {}
};

#endif // AGENT_H