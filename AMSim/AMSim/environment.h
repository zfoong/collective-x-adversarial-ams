#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <vector>

enum MatterType {
	collective,
	adversarial,
	learner,
	teacher
};

const float RADIUS = 1;
const float MASS = 100;
const float V = 1;

struct Matter {
	float pos[2];
	float pos_aw[2];
	int posMultiplier[2] = {0}; // Global position multiplier
	float r = RADIUS; // radius, setting to const for now
	float v = V;
	float m = MASS; // mass of matter
	float ort[2];
	int neighbourCount = 0;
	MatterType type = teacher;
};

class Environment {
public:
	Environment(float = 1, float = 32, bool=true);
	std::vector<Matter> prevMatters;
	std::vector<Matter> matters;
	std::vector<float> ReturnState();
	std::vector<float> Step(std::vector<float>, std::vector<float>&, std::vector<float>&);
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