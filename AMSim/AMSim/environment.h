#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <vector>

enum MatterType {
	collective,
	adversarial
};

const float RADIUS = 1;
const float MASS = 100;
const float V = 1;

struct Matter {
	float x; // x pos
	float y; // y pos
	float r = RADIUS; // radius, setting to const for now
	float v = V;
	float m = MASS; // mass of matter
	float ort[2];
	int neighbourCount = 0;
	MatterType type = collective;
};

class Environment {
public:
	Environment();
	std::vector<Matter> prevMatters;
	std::vector<Matter> matters;
	std::vector<float> ReturnState();
	std::vector<float> Step(std::vector<float>, std::vector<float>&);
	void Movement(Matter&, float);
	void Display();
	virtual ~Environment();
protected:
private:
	//Environment(const Environment& other) {}
	Environment& operator=(const Environment& other) {}
	void AddMatter(float, float);
	void RemoveMatters();
	float ClipToScreen(float);
};

#endif // ENVIRONMENT_H