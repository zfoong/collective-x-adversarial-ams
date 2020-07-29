#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <vector>

enum MatterType {
	collective,
	adversarial
};

const float RADIUS = 1;
const float V = 1;

struct Matter {
	float pos[2];
	int posMultiplier[2] = {0}; // Global position multiplier
	float r = RADIUS; // radius, setting to const for now
	float v = V;
	float ort[2];
	int neighbourCount = 0;
	float acmlActiveWork = 0;
	float acmlCurrentActiveWork = 0;
	MatterType type = adversarial;
};

class Environment {
public:
	Environment(float = 1, float = 32, bool=true);
	std::vector<Matter> prevMatters;
	std::vector<Matter> matters;
	float t = 0;
	std::vector<float> ReturnState();
	std::vector<float> ReturnCState();
	std::vector<float> ReturnAllState();
	std::vector<float> Step(std::vector<float>, std::vector<float>&, bool&);
	float Environment::returnActiveWork();
	float Environment::returnCurrentActiveWork();
	void Movement(Matter&, float);
	void Display();
	virtual ~Environment();
protected:
private:
	//Environment(const Environment& other) {}
	// Environment& operator=(const Environment& other) {}
	void AddMatter(MatterType);
	void AddMatter(MatterType, float, float, float, float);
	void RemoveMatters();
};

#endif // ENVIRONMENT_H