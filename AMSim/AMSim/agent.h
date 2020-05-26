#ifndef AGENT_H
#define AGENT_H

#include <string>

class Agent {
public:
	Agent();
	float QTable[16][16];
	void UpdateQTable();
	float ReturnAction();
	void SaveQTable();
	void LoadQTable();
	virtual ~Agent();
protected:
	int UnpackState2Index();
private:
	Agent(const Agent& other) {}
	Agent& operator=(const Agent& other) {}
};

#endif // AGENT_H