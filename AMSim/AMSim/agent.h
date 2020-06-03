#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <vector>

class Agent {
public:
	Agent();
	float QTable[16][16] = {{0}};
	void UpdateQTable(float, int, float, float);
	void UpdateQTable(std::vector<float>, std::vector<int>, std::vector<float>, std::vector<float>);
	float ReturnAction(float, int&);
	std::vector<float> ReturnAction(std::vector<float>, std::vector<int>&);
	void SaveQTable();
	void LoadQTable();
	virtual ~Agent();
protected:
private:
	//Agent(const Agent& other) {}
	Agent& operator=(const Agent& other) {}
};

#endif // AGENT_H