#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <vector>

enum MatterType {
	flocking,
	clustering
};

const float RADIUS = 1;
const float V = 1;

// properties of active matter, particles, or SPP, or agents as referred to the thesis
struct Matter {
	float pos[2]; // position r
	int posMultiplier[2] = {0}; // global position multiplier
	float r = RADIUS; // radius, setting to const for now
	float v = V; // self-propelled speed
	float ort[2]; // orientation theta
	float acmlActiveWork = 0; // storing total active work
	float acmlCurrentActiveWork = 0; // storing current active work
	MatterType type = clustering;
};

class Environment {
public:
	Environment(float = 1, float = 32, bool=true);
	std::vector<Matter> prevMatters;
	std::vector<Matter> matters;
	float t = 0;
	std::vector<float> ReturnFState();
	std::vector<float> ReturnCState();
	std::vector<float> ReturnAllState();
	std::vector<float> Step(std::vector<float>, std::vector<float>&, bool&);
	float Environment::returnAllActiveWork();
	float Environment::returnActiveWork_f();
	float Environment::returnActiveWork_c();
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