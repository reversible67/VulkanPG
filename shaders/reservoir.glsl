#define RESERVOIR_SIZE 1

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

struct PackedReservoir
{
	uint sampleSeed;
	float16_t W;
	uint M;
};

struct PackedIndirectReservoir
{
	vec3 position;
	f16vec3 normal;
	f16vec3 radiance;
	float16_t W;
	uint M;
};

struct Sample 
{
	uint sampleSeed;
	float sumWeights; //总权重
	float W;
};

struct Reservoir
{
	Sample samples[RESERVOIR_SIZE];
	uint M;	//样本数量
};

struct IndirectReservoir
{
	vec3 position;
	vec3 normal;
	vec3 radiance;
	float sumWeights;
	float W;	//RIS
	uint M;	//样本数量
};

struct IncidentRadianceReservoir
{
	uint index;
	float sumWeights;
	float W;
	uint M;
	float weight;
};

bool updateReservoir(inout Reservoir reservoir, int i, float weight, uint sampleSeed, inout uint seed)
{
	if (weight > 0.0)
	{
		reservoir.samples[i].sumWeights += weight;
		float replacePossibility = weight / reservoir.samples[i].sumWeights;
		if (rnd(seed) < replacePossibility)
		{
			reservoir.samples[i].sampleSeed = sampleSeed;
			return true;
		}
	}
	return false;
}

bool updateIndirectReservoir(inout IndirectReservoir reservoir, float weight, IndirectReservoir other, inout uint seed)
{
	if (weight > 0.0)
	{
		reservoir.sumWeights += weight;
		float replacePossibility = weight / reservoir.sumWeights;
		if (rnd(seed) < replacePossibility)
		{
			reservoir.position = other.position;
			reservoir.normal = other.normal;
			reservoir.radiance = other.radiance;
			return true;
		}
	}
	return false;
}

void updateIncidentRadianceReservoir(inout IncidentRadianceReservoir reservoir, float weight, uint index, inout uint seed)
{
	reservoir.sumWeights += weight;
	float replacePossibility = weight / reservoir.sumWeights;
	if (rnd(seed) < replacePossibility)
	{
		reservoir.index = index;
		reservoir.weight = weight;
	}
}

bool combineReservoirs(inout Reservoir self, Reservoir other, float pHat[RESERVOIR_SIZE], inout uint seed)
{
	bool updated = false;
	for (int i = 0; i < RESERVOIR_SIZE; ++i)
	{
		float weight = pHat[i] * other.samples[i].W * other.M;
		if (updateReservoir(self, i, weight, other.samples[i].sampleSeed, seed))
			updated = true;
	}
	self.M += other.M;
	return updated;
}//merge函数

bool combineIndirectReservoirs(inout IndirectReservoir self, IndirectReservoir other, float pHat, inout uint seed)
{
	float weight = pHat * other.W * other.M;
	self.M += other.M;
	return updateIndirectReservoir(self, weight, other, seed);
}

Reservoir createReservoir()
{
	Reservoir reservoir;
	for (int i = 0; i < RESERVOIR_SIZE; ++i)
		reservoir.samples[i].sumWeights = 0.0;
	reservoir.M = 0;
	return reservoir;
}

IndirectReservoir createIndirectReservoir()
{
	IndirectReservoir reservoir;
	reservoir.sumWeights = 0.0;
	reservoir.M = 0;
	reservoir.radiance = vec3(0.0);
	reservoir.position = vec3(0.0);
	return reservoir;
}

float luminance(vec3 rgb)
{
	return 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
}

IncidentRadianceReservoir createIncidentRadianceReservoir()
{
	IncidentRadianceReservoir reservoir;
	reservoir.sumWeights = 0.0;
	reservoir.M = 0;
	reservoir.W = 1.0;
	return reservoir;
}