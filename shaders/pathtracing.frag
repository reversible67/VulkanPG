#version 460

#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
#include "reservoir.glsl"
#include "gmm.glsl"
#include "concentric.glsl"

const int BIAS_CORRECTION_RAY_TRACED_DI = 1;
const int BIAS_CORRECTION_RAY_TRACED_GI = 1;
const int SPATIAL_RIS_WITH_MIS_DI = 1;
const int SPATIAL_RIS_WITH_MIS_GI = 1;

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inDir;

layout (location = 0) out vec4 outFragColor;
layout (location = 1) out uint outHistory;

layout (binding = 0) uniform sampler2D samplerPosition;
layout (binding = 1) uniform sampler2D samplerNormal;
layout (binding = 2) uniform sampler2D samplerWhiteNoise;
layout (binding = 3) uniform sampler2D samplerBlueNoise;
layout (binding = 4) uniform sampler2D samplerMotion;
layout (binding = 5) uniform usampler2D samplerHistory;
layout (binding = 6) uniform sampler2D samplerEnvMap;
layout (binding = 7) uniform sampler2D samplerEnvHDRCache;
layout (binding = 8) uniform sampler2D samplerPreviousDepth;
layout (binding = 9) uniform sampler2D samplerPreviousColor;
layout (binding = 11) uniform sampler2D samplerAlbedo;
layout (binding = 12) uniform sampler2D samplerSpecular;
layout (binding = 13) uniform sampler2D samplerImages[];
layout (binding = 14, scalar) buffer readonly SceneDescs { SceneDesc sceneDescs[]; };
//layout (binding = 15, scalar) buffer EnvAccels { EnvAccel envAccel[]; };
//layout (binding = 16) uniform sampler2D samplerEnvPdfs;
layout (binding = 17, scalar) buffer Reservoirs { PackedReservoir reservoirs[]; };
layout (binding = 18, scalar) readonly buffer PreviousReservoirs { PackedReservoir previousReservoirs[]; };
layout (binding = 19, scalar) buffer IndirectReservoirs { PackedIndirectReservoir indirectReservoirs[]; };
layout (binding = 20, scalar) readonly buffer PreviousIndirectReservoirs { PackedIndirectReservoir previousIndirectReservoirs[]; };
layout (binding = 21, scalar) buffer IncidentRadianceGrid { IncidentRadianceGridCell incidentRadianceGridCells[]; };
layout (binding = 22, scalar) buffer GMMStatisticsPack0 { vec4 gmmStatisticsPack0[]; };
layout (binding = 23, scalar) buffer GMMStatisticsPack1 { vec4 gmmStatisticsPack1[]; };
layout (binding = 24, scalar) buffer readonly GMMStatisticsPack0Prev { vec4 gmmStatisticsPack0Prev[]; };
layout (binding = 25, scalar) buffer readonly GMMStatisticsPack1Prev { vec4 gmmStatisticsPack1Prev[]; };
layout (binding = 26, scalar) buffer writeonly VPLs { vec4 vpls[]; };
layout (binding = 27) uniform accelerationStructureEXT topLevelAS;
layout (binding = 28, scalar) buffer BoundingVoxels { BoundingVoxel boundingVoxels[]; };

layout (binding = 10, scalar) uniform UBO 
{
	mat4 inverseMVP;
	mat4 projection;
	vec3 viewPos;
	int displayDebugTarget;
	int historyLength;
	float exposure;
	float alphaCutoff;
	int frameNumber;
	int sampleCount;
	mat3 envRot;
	int numBounces;
	float probability;
	int numCandidates;
	int temporalSamplesDI;
	int temporalSamplesGI;
	int reference;
	vec3 spotDir;
	vec3 spotLightPos;
	float spotAngle;
	float spotLightIntensity;
	int spatialReuseRadiusDI;
	int spatialReuseRadiusGI;
	int spatialSamplesDI;
	int spatialSamplesGI;
	int russianRoulette;
	float clampValue;
} ubo;

layout (push_constant, scalar) uniform PushConstants
{
	int modelIndex;
	vec3 gridBegin;
	ivec3 gridDim;
	float cellSize;
	int lobeCount;
};

layout (buffer_reference, scalar) buffer readonly Vertices { Vertex v[]; };
layout (buffer_reference, scalar) buffer readonly Indices { ivec3 i[]; };
layout (buffer_reference, scalar) buffer Materials { Material m[]; }; // Array of all materials on an object
layout (buffer_reference, scalar) buffer MaterialIndices { int i[]; }; // Material ID for each triangle
layout (buffer_reference, scalar) buffer FirstPrimitives { int p[]; };

uint rayFlags;

struct PathVertex
{
	vec3 position;
	vec3 direction;
	vec3 throughput;
};

vec3 sampleEnvMap(vec3 dir)
{
	return textureLod(samplerEnvMap, sphere2UV(dir), 0).rgb;//vec3(3.0);//
}

// 采样预计算的 HDR cache
vec3 sampleHDR(vec2 xi)
{
    vec2 xy = textureLod(samplerEnvHDRCache, xi, 0).rg; // x, y

    // 获取角度
    float phi = 2.0 * M_PI * (xy.x - 0.5);    // [-pi ~ pi]
    float theta = M_PI * (xy.y - 0.5);        // [-pi/2 ~ pi/2]   

    // 球坐标计算方向
    vec3 L = vec3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));

    return L;
}

// 输入光线方向 L 获取 HDR 在该位置的概率密度
float hdrPdf(vec3 dir)
{
    vec2 uv = sphere2UV(dir);   // 方向向量转 uv 纹理坐标

    float pdf = textureLod(samplerEnvHDRCache, uv, 0).b;      // 采样概率密度
    float theta = M_PI * (0.5 - uv.y);            // theta 范围 [-pi/2 ~ pi/2]
    float cos_theta = max(cos(theta), 1e-10);

	ivec2 hdrResolution = textureSize(samplerEnvHDRCache, 0);

    // 球坐标和图片积分域的转换系数
	float p_convert = float(hdrResolution.x * hdrResolution.y) / (2.0 * M_PI * M_PI * cos_theta);  
    
	return pdf * p_convert;
}

vec3 getDir(inout float pdf, inout uint seed, vec3 normal, vec3 tangent, vec3 bitangent, mat3 invEnvRot)
{	
	vec2 uv = vec2(rnd(seed), rnd(seed));
	vec3 dir;
	if (ENV_MAP_IS == 1)
	{
		vec3 envDir = sampleHDR(uv);
		pdf = hdrPdf(envDir);
		dir = invEnvRot * envDir;
	}
	else
	{
		dir = hemisphereSample_uniform(uv);
		dir = dir.x * tangent + dir.y * bitangent + dir.z * normal;
	}
	return dir;
}

vec3 sampleEnvironmentMap(inout float pdf, inout uint seed, vec3 normal, vec3 tangent, vec3 bitangent, mat3 invEnvRot)
{	
	vec2 uv = vec2(rnd(seed), rnd(seed));
	vec3 dir;
	if (ENV_MAP_IS == 1)
	{
		vec3 envDir = sampleHDR(uv);
		dir = invEnvRot * envDir;
		pdf = hdrPdf(envDir);
	}
	else
	{
		dir = hemisphereSample_uniform(uv);
		dir = dir.x * tangent + dir.y * bitangent + dir.z * normal;
		pdf = 1 / (2 * M_PI);
	}
	return dir;
}

float pdfEnvironmentMap(vec3 dir, vec3 normal)
{
	if (ENV_MAP_IS == 1)
		return hdrPdf(ubo.envRot * dir);
	else
		return 1 / (2 * M_PI);
}

float evaluatePHat(vec3 dir, HitMaterial material, vec3 view, vec3 normal, mat3 invEnvRot)
{
	float brdfPdf = pdfBRDF(view, normal, dir, material);
	if (brdfPdf <= 0.0)
		return 0.0;
	vec3 brdf = evaluateBRDF(view, normal, dir, material);
	//对比glossy材质是否考虑brdf和brdfPdf
	return max(0.0, dot(normal, dir)) * luminance(brdf * sampleEnvMap(ubo.envRot * dir)) / brdfPdf;
}

float evaluatePHat(inout float pdf, inout uint seed, HitMaterial material, vec3 view, vec3 normal, vec3 tangent, vec3 bitangent, mat3 invEnvRot)
{
	vec3 dir = getDir(pdf, seed, normal, tangent, bitangent, invEnvRot);
	return evaluatePHat(dir, material, view, normal, invEnvRot);
}

void traceTransparency(rayQueryEXT rayQuery, inout uint seed)
{
	while (rayQueryProceedEXT(rayQuery))
	{
		if (rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
		{
			bool transparent = false;
			if (ALPHA_TEST == 1)
			{
				uint instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
				SceneDesc sceneDesc = sceneDescs[instanceCustomIndex];
				MaterialIndices materialIndices = MaterialIndices(sceneDesc.materialIndexAddress);
				Materials materials = Materials(sceneDesc.materialAddress);
				Indices indices = Indices(sceneDesc.indexAddress);
				Vertices vertices = Vertices(sceneDesc.vertexAddress);
				FirstPrimitives firstPrimitives = FirstPrimitives(sceneDesc.firstPrimitiveAddress);
				int geometryIndex = rayQueryGetIntersectionGeometryIndexEXT(rayQuery, false);
				int primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false) + firstPrimitives.p[geometryIndex];
				ivec3 index = indices.i[primitiveIndex];
				Vertex v0 = vertices.v[index.x];
				Vertex v1 = vertices.v[index.y];
				Vertex v2 = vertices.v[index.z];
				vec2 attribs = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
				const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs);
				int materialIndex = materialIndices.i[primitiveIndex];
				if (materialIndex >= 0)
				{
					Material material = materials.m[materialIndex];
					float alphaValue = material.baseColor.a;
					if (material.baseColorImage >= 0)
					{
						uint baseColorImage = material.baseColorImage + sceneDesc.imageOffset;
						vec2 texCoord = v0.uv * barycentrics.x + v1.uv * barycentrics.y + v2.uv * barycentrics.z;
						alphaValue = texture(samplerImages[nonuniformEXT(baseColorImage)], texCoord).a;
					}
					if (material.alphaMode == 1 || material.alphaMode == 2)
						if (alphaValue < ubo.alphaCutoff)
							transparent = true;
					//if (material.alphaMode == 2)
					//	if (alphaValue < rnd(seed))
					//		transparent = true;
					//if (material.transmission > 0.001)
					//	transparent = true;
				}
			}
			if (!transparent)
				rayQueryConfirmIntersectionEXT(rayQuery);
		}
	}
}

vec2 randomChooseNeighbor(uint seed, float radius)
{
	float x = 2 * radius * rnd(seed) - radius;
	float y = 2 * radius * rnd(seed) - radius;
	return vec2(x, y);
}

bool calcGeoSimilarity(vec2 inUV, vec2 rnd_UV)
{
	//角度比较
	float maxAngle = 15.0; // 最大夹角，单位为度
	vec3 normal1 =  texture(samplerNormal, inUV).xyz;
	vec3 normal2 =  texture(samplerNormal, rnd_UV).xyz;
	float dotProduct = dot(normalize(normal1), normalize(normal2));
	float angle = degrees(acos(clamp(dotProduct, -1.0, 1.0)));
	bool angleWithinThreshold = (angle <= maxAngle); // 判断夹角是否在阈值范围内
	
	//深度比较
	float depth1 =  texture(samplerPosition, inUV).w;
	float depth2 =  texture(samplerPosition, rnd_UV).w;
	float depthDiff = abs(depth1 - depth2); // 深度差异
	bool depthWithinThreshold = (depthDiff <= depth1 * 0.2); // 判断深度差异是否在阈值范围内
	
	if(angleWithinThreshold == true &&  depthWithinThreshold ==true){
		return true;
	}else{
		return false;
	}
}

bool RAB_ValidateGISampleWithJacobian(inout float jacobian)
{
    // Sold angle ratio is too different. Discard the sample.
    if (jacobian > 10.0 || jacobian < 1 / 10.0) {
        return false;
    }

    // clamp Jacobian.
    jacobian = clamp(jacobian, 1 / 3.0, 3.0);

    return true;
}

// Assume normalized input. Output is on [-1, 1] for each component.
vec2 unitVectorToOctahedron(vec3 v)
{
    // Project the sphere onto the octahedron, and then onto the xy plane
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
}

vec3 octahedronToUnitVector(vec2 e)
{
    vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    return normalize(v);
}

float computeSphericalExcess(vec3 A, vec3 B, vec3 C)
{
    float cosAB = dot(A, B);
    float sinAB = 1.0 - cosAB * cosAB;
    float cosBC = dot(B, C);
    float sinBC = 1.0 - cosBC * cosBC;
    float cosCA = dot(C, A);
    float cosC = cosCA - cosAB * cosBC;
    float sinC = sqrt(sinAB * sinBC - cosC * cosC);
	if (isnan(sinC))
		sinC = 0.0;
    float inv = (1.0 - cosAB) * (1.0 - cosBC);
	return 2.0 * atan(sinC, sqrt((sinAB * sinBC * (1.0 + cosBC) * (1.0 + cosAB)) / inv) + cosC);
}

float octahedralSolidAngle(vec2 coord, float invResolution)
{
	vec3 dir1 = octahedronToUnitVector(coord + vec2(1.0, -1.0) * invResolution);
	vec3 dir2 = octahedronToUnitVector(coord + vec2(-1.0, 1.0) * invResolution);

	float solidAngle1 = computeSphericalExcess(octahedronToUnitVector(coord + vec2(-1.0) * invResolution), dir1, dir2);
	float solidAngle2 = computeSphericalExcess(octahedronToUnitVector(coord + vec2(1.0) * invResolution), dir1, dir2);

	return solidAngle1 + solidAngle2;
}

int getGridIndex(vec3 position)
{
	ivec3 gridCell = ivec3((position - gridBegin) / cellSize);
	if (HASHING == 0)
	{
		if (all(greaterThanEqual(gridCell, ivec3(0))) && all(lessThan(gridCell, ivec3(gridDim))))
			return gridCell.z * gridDim.x * gridDim.y + gridCell.y * gridDim.x + gridCell.x;
		else
			return -1;
	}
	else
		return (73856093 * gridCell.x + 19349669 * gridCell.y + 83492791 * gridCell.z) % (gridDim.x * gridDim.y * gridDim.z);
}

void updateRadianceField(vec3 position, vec3 direction, vec3 radiance)
{
	int gridIndex = getGridIndex(position);
	if (gridIndex >= 0)
	{
		ivec2 coord = clamp(ivec2(floor(INCIDENT_RADIANCE_MAP_SIZE * (unitVectorToOctahedron(direction) * 0.5 + vec2(0.5)))), ivec2(0), ivec2(INCIDENT_RADIANCE_MAP_SIZE - 1));
		atomicAdd(incidentRadianceGridCells[gridIndex].incidentRadianceSum[coord.y * INCIDENT_RADIANCE_MAP_SIZE + coord.x], luminance(radiance));
		atomicAdd(incidentRadianceGridCells[gridIndex].incidentRadianceCount[coord.y * INCIDENT_RADIANCE_MAP_SIZE + coord.x], 1);
	}
}

// Forward declaration for readHit (needed by sampleIntraVoxel)
void readHit(rayQueryEXT rayQuery, inout vec3 hitPos, inout vec3 hitNormal, inout HitMaterial hitMaterial, vec3 dir);

// VXPG: Update bounding voxel with hit position and irradiance
void updateBoundingVoxel(vec3 position, vec3 irradiance)
{
	int gridIndex = getGridIndex(position);
	if (gridIndex >= 0)
	{
		// Update AABB atomically using compare-and-swap on uint representation
		uint posX = floatBitsToUint(position.x);
		uint posY = floatBitsToUint(position.y);
		uint posZ = floatBitsToUint(position.z);
		
		uint oldMinX = atomicMin(boundingVoxels[gridIndex].aabbMinX, posX);
		uint oldMinY = atomicMin(boundingVoxels[gridIndex].aabbMinY, posY);
		uint oldMinZ = atomicMin(boundingVoxels[gridIndex].aabbMinZ, posZ);
		uint oldMaxX = atomicMax(boundingVoxels[gridIndex].aabbMaxX, posX);
		uint oldMaxY = atomicMax(boundingVoxels[gridIndex].aabbMaxY, posY);
		uint oldMaxZ = atomicMax(boundingVoxels[gridIndex].aabbMaxZ, posZ);
		
		// Update irradiance
		atomicAdd(boundingVoxels[gridIndex].totalIrradiance, luminance(irradiance));
		atomicAdd(boundingVoxels[gridIndex].sampleCount, 1);
	}
}

// VXPG: Sample a voxel based on weighted contribution
int sampleVoxel(vec3 shadingPos, vec3 normal, inout uint seed, out float pdf)
{
	const int numSamples = 8; // Sample a subset of voxels
	float weights[numSamples];
	int indices[numSamples];
	float totalWeight = 0.0;
	
	// Gather candidate voxels
	for (int i = 0; i < numSamples; ++i)
	{
		// Random voxel selection (could use spatial hashing or clustering)
		int voxelIdx = int(rnd(seed) * float(gridDim.x * gridDim.y * gridDim.z));
		indices[i] = voxelIdx;
		
		if (boundingVoxels[voxelIdx].sampleCount == 0)
		{
			weights[i] = 0.0;
			continue;
		}
		
		// Calculate voxel center
		int x = voxelIdx % gridDim.x;
		int y = (voxelIdx / gridDim.x) % gridDim.y;
		int z = voxelIdx / (gridDim.x * gridDim.y);
		vec3 voxelCenter = gridBegin + (vec3(x, y, z) + 0.5) * cellSize;
		
		// Estimate contribution: irradiance × visibility × BSDF approximation
		vec3 toVoxel = voxelCenter - shadingPos;
		float dist = length(toVoxel);
		vec3 dir = toVoxel / max(dist, 0.001);
		
		// Simple visibility check (1 if facing, 0 otherwise)
		float visibility = max(0.0, dot(normal, dir));
		
		// Weight by irradiance and geometric term
		float irradiance = boundingVoxels[voxelIdx].totalIrradiance;
		weights[i] = irradiance * visibility / max(dist * dist, 0.01);
		totalWeight += weights[i];
	}
	
	if (totalWeight < 1e-6)
	{
		pdf = 0.0;
		return -1;
	}
	
	// Sample voxel based on weights
	float r = rnd(seed) * totalWeight;
	float cumulative = 0.0;
	for (int i = 0; i < numSamples; ++i)
	{
		cumulative += weights[i];
		if (r <= cumulative)
		{
			pdf = weights[i] / totalWeight;
			return indices[i];
		}
	}
	
	pdf = weights[numSamples - 1] / totalWeight;
	return indices[numSamples - 1];
}

// VXPG: Sample a point within the voxel's AABB and trace to geometry
bool sampleIntraVoxel(int voxelIdx, inout uint seed, out vec3 sampledPos, out vec3 sampledNormal, out HitMaterial sampledMaterial)
{
	vec3 aabbMin = vec3(
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMinX),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMinY),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMinZ)
	);
	vec3 aabbMax = vec3(
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMaxX),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMaxY),
		uintBitsToFloat(boundingVoxels[voxelIdx].aabbMaxZ)
	);
	
	// Check if AABB is valid
	if (any(greaterThan(aabbMin, aabbMax)))
		return false;
	
	// Sample a point uniformly in the AABB
	vec3 samplePoint = aabbMin + vec3(rnd(seed), rnd(seed), rnd(seed)) * (aabbMax - aabbMin);
	
	// Sample a direction from the AABB center to test for geometry
	vec3 aabbCenter = (aabbMin + aabbMax) * 0.5;
	vec3 toSample = samplePoint - aabbCenter;
	float rayDist = length(toSample);
	vec3 rayDir = toSample / max(rayDist, 0.001);
	
	// Trace a ray to see if we hit geometry
	rayQueryEXT rayQuery;
	rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, aabbCenter, 0.001, rayDir, rayDist);
	while(rayQueryProceedEXT(rayQuery)) {}
	
	if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
	{
		// Use temporary variables for readHit (readHit requires inout, but our outputs are declared as out)
		vec3 tempPos = vec3(0.0);
		vec3 tempNormal = vec3(0.0, 0.0, 1.0);
		HitMaterial tempMaterial;
		tempMaterial.baseColor = vec3(1.0);
		tempMaterial.metallic = 0.0;
		tempMaterial.specular = 0.5;
		tempMaterial.roughness = 1.0;
		tempMaterial.metallicRoughness = true;
		tempMaterial.specularColor = vec3(0.0);
		
		readHit(rayQuery, tempPos, tempNormal, tempMaterial, rayDir);
		
		// Assign to output parameters
		sampledPos = tempPos;
		sampledNormal = tempNormal;
		sampledMaterial = tempMaterial;
		return true;
	}
	
	return false;
}

IndirectReservoir readPreviousIndirectReservoir(int index)
{
	IndirectReservoir reservoir = createIndirectReservoir();
	reservoir.position = previousIndirectReservoirs[index].position;
	reservoir.normal = previousIndirectReservoirs[index].normal;
	reservoir.M = previousIndirectReservoirs[index].M;
	reservoir.W = previousIndirectReservoirs[index].W;
	reservoir.radiance = previousIndirectReservoirs[index].radiance;
	return reservoir;
}

void writeIndirectReservoir(int index, IndirectReservoir reservoir)
{
	//position加不加f16vec3没区别
	indirectReservoirs[index].position = reservoir.position;
	indirectReservoirs[index].normal = f16vec3(reservoir.normal);
	indirectReservoirs[index].M = reservoir.M;
	indirectReservoirs[index].W = float16_t(reservoir.W);
	indirectReservoirs[index].radiance = f16vec3(reservoir.radiance);
}

Reservoir readPreviousReservoir(int index)
{	
	Reservoir reservoir = createReservoir();
	reservoir.samples[0].sampleSeed = previousReservoirs[index].sampleSeed;
	reservoir.M = previousReservoirs[index].M;
	reservoir.samples[0].W = previousReservoirs[index].W;
	return reservoir;
}

void writeReservoir(int index, Reservoir reservoir)
{	
	reservoirs[index].sampleSeed = reservoir.samples[0].sampleSeed;
	reservoirs[index].M = reservoir.M;
	reservoirs[index].W = float16_t(reservoir.samples[0].W);
}

HitMaterial readMaterial(vec2 uv)
{
	HitMaterial hitMaterial;
	vec3 albedo = texture(samplerAlbedo, uv).rgb;
	vec4 specular = texture(samplerSpecular, uv);
	hitMaterial.baseColor = albedo;
	hitMaterial.specular = 0.5;
	if (specular.a > 0.5)
	{
		hitMaterial.metallic = specular.r;
		hitMaterial.metallicRoughness = true;
	}
	else
	{
		hitMaterial.specularColor = specular.rgb;
		hitMaterial.metallicRoughness = false;
	}
	hitMaterial.roughness = texture(samplerMotion, uv).w;
	return hitMaterial;
}

void readHit(rayQueryEXT rayQuery, inout vec3 hitPos, inout vec3 hitNormal, inout HitMaterial hitMaterial, vec3 dir)
{
	uint instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
	SceneDesc sceneDesc = sceneDescs[instanceCustomIndex];
	MaterialIndices materialIndices  = MaterialIndices(sceneDesc.materialIndexAddress);
	Materials materials = Materials(sceneDesc.materialAddress);
	Indices indices = Indices(sceneDesc.indexAddress);
	Vertices vertices = Vertices(sceneDesc.vertexAddress);
	FirstPrimitives firstPrimitives = FirstPrimitives(sceneDesc.firstPrimitiveAddress);
	int geometryIndex = rayQueryGetIntersectionGeometryIndexEXT(rayQuery, true);
	int primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true) + firstPrimitives.p[geometryIndex];
	ivec3 index = indices.i[primitiveIndex];

	Vertex v0 = vertices.v[index.x];
	Vertex v1 = vertices.v[index.y];
	Vertex v2 = vertices.v[index.z];

	vec2 attribs = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
	// Interpolate normal
	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs);
	hitPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
	hitNormal = normalize(v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z);
	if (dot(hitNormal, dir) > 0.0)
		hitNormal = -hitNormal;
	vec4 tangent = v0.tangent * barycentrics.x + v1.tangent * barycentrics.y + v2.tangent * barycentrics.z;
	tangent.xyz = normalize(tangent.xyz);
				
	vec3 albedo = vec3(1.0);
	vec2 metallicRoughness = vec2(0.0, 1.0);
	vec4 specularGlossiness = vec4(0.0);
	// Material of the object
	int materialIndex = materialIndices.i[primitiveIndex];
	if (materialIndex >= 0)
	{
		Material material = materials.m[materialIndex];
		vec2 texCoord = v0.uv * barycentrics.x + v1.uv * barycentrics.y + v2.uv * barycentrics.z;
		if (BUMPED == 1 && material.normalImage >= 0)
		{
			uint normalImage = material.normalImage + sceneDesc.imageOffset;
			hitNormal = applyNormalMap(hitNormal, tangent, texture(samplerImages[nonuniformEXT(normalImage)], texCoord).xyz);
		}
		albedo = material.baseColor.rgb;
		if (material.baseColorImage >= 0)
		{
			uint baseColorImage = material.baseColorImage + sceneDesc.imageOffset;
			albedo *= pow(texture(samplerImages[nonuniformEXT(baseColorImage)], texCoord).rgb, vec3(2.2));
		}
		hitMaterial.baseColor = albedo;
		if (material.metallicRoughness)
			metallicRoughness = vec2(material.metallic, material.roughness);
		else
			specularGlossiness = vec4(material.specularFactor, material.roughness);
		if (material.metallicRoughnessImage >= 0)
		{
			uint metallicRoughnessImage = material.metallicRoughnessImage + sceneDesc.imageOffset;
			if (material.metallicRoughness)
				metallicRoughness *= texture(samplerImages[nonuniformEXT(metallicRoughnessImage)], texCoord).bg;
			else
			{
				vec4 specularGlossinessTexel = texture(samplerImages[nonuniformEXT(metallicRoughnessImage)], texCoord);
				specularGlossinessTexel.rgb = pow(specularGlossinessTexel.rgb, vec3(2.2));
				specularGlossiness.rgb *= specularGlossinessTexel.rgb;
				specularGlossiness.a *= specularGlossinessTexel.a;
			}
		}
		hitMaterial.metallicRoughness = material.metallicRoughness;
		if (material.metallicRoughness)
		{
			hitMaterial.metallic = metallicRoughness.r;
			hitMaterial.roughness = metallicRoughness.g;
		}
		else
		{
			hitMaterial.specularColor = specularGlossiness.rgb;
			hitMaterial.roughness = 1.0 - specularGlossiness.a * specularGlossiness.a;
		}
	}
}

void main()
{
	//gl_RayFlagsSkipClosestHitShaderEXT也可以
	rayFlags = gl_RayFlagsOpaqueEXT;

	// Get G-Buffer values
	vec4 positionTexel = texture(samplerPosition, inUV);
	float depth = positionTexel.w;
	vec3 fragPos = positionTexel.xyz;
	vec3 normal = texture(samplerNormal, inUV).xyz;
	vec4 motionTexel = texture(samplerMotion, inUV);

	uvec2 resolution = textureSize(samplerNormal, 0);
	int pixel = int(resolution.x * uint(resolution.y * inUV.y) + resolution.x * inUV.x);
	int lobeIndex = pixel * lobeCount;

	if (length(normal) < 0.1)
	{
		outFragColor = vec4(0.0);
		if (SSPG == 1)
		{
			if (SGM == 1)
			{
				gmmStatisticsPack0[pixel] = vec4(0.0);
				gmmStatisticsPack1[pixel] = vec4(0.0);
			}
			else
			{
				vec4 weights = vec4(0.1, 0.2, 0.3, 0.4);
				for (int i = 0; i < lobeCount; ++i)
				{
					vec2 uv = 1.0 + 1.0 * vec2(i / 2, i % 2);
					gmmStatisticsPack0[lobeIndex + i] = vec4(uv, 0.01, 0.01);
					gmmStatisticsPack1[lobeIndex + i] = vec4(0.0, 1.0, 0.0, weights[0]);
				}
			}
		}
		return;
	}

	vec3 motionVector = motionTexel.xyz;
	vec2 previousUV = inUV - 0.5 * motionVector.xy;
	uint history = texture(samplerHistory, previousUV).x;
	float currentFactor = 1.0 / float(history + 1);
	//float currentFactor = 0.05;
	if (ubo.reference == 1)
		currentFactor = 1.0 / float(ubo.frameNumber + 1);
	bool reprojected = true;
	if (previousUV.x < 0.0 || previousUV.x > 1.0 || previousUV.y < 0.0 || previousUV.y > 1.0)
	{
		currentFactor = 1.0;
		outHistory = 0;
		reprojected = false;
	}
	else if (abs(texture(samplerPreviousDepth, previousUV).x - depth + motionVector.z) > depth * 0.2)
	{
		currentFactor = 1.0;
		outHistory = 0;
	}
	else if (ubo.reference == 0)
		outHistory = min(history + 1, ubo.historyLength);
	
	int previousPixel = int(resolution.x * uint(resolution.y * previousUV.y) + resolution.x * previousUV.x);

	// Initialize the random number
	uint seed = tea(uint(1920 * gl_FragCoord.y + gl_FragCoord.x), ubo.frameNumber);
	vec2 noise = texelFetch(samplerWhiteNoise, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0).rg * M_PI * 2.0;
	vec2 blueNoise = texelFetch(samplerBlueNoise, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0).rg;

	// Move origin slightly away from the surface to avoid self-occlusion
	vec3 origin = fragPos + 0.00001 * normal;
	vec3 direction;

	// Finding the basis (tangent and bitangent) from the normal
	vec3 tangent, bitangent;
	computeDefaultBasis(normal, tangent, bitangent);
	vec2 uv;
	mat3 invEnvRot = inverse(ubo.envRot);

	vec3 view = -normalize(inDir);
	HitMaterial hitMaterial;
	vec3 albedo = texture(samplerAlbedo, inUV).rgb;
	vec4 specular = texture(samplerSpecular, inUV);
	hitMaterial.baseColor = albedo;
	hitMaterial.specular = 0.5;
	if (specular.a > 0.5)
	{
		hitMaterial.metallic = specular.r;
		hitMaterial.metallicRoughness = true;
	}
	else
	{
		hitMaterial.specularColor = specular.rgb;
		hitMaterial.metallicRoughness = false;
	}
	hitMaterial.roughness = motionTexel.w;
	HitMaterial primaryMaterial = hitMaterial;

	rayQueryEXT rayQuery;
	Reservoir reservoir = createReservoir();
	float pHat = 1.0;
	if (RESTIR_DI == 1 && NEE == 1)
	{
		vec3 dir;
		//最后算出的W有1/pdf这一项，所以必须准确
		float pdf = 1 / (2 * M_PI);//1 / (2 * pi) * pi
		for (int i = 0; i < ubo.numCandidates; ++i)
		{
			uint sampleSeed = seed;
			float tempPHat = evaluatePHat(pdf, seed, hitMaterial, view, normal, tangent, bitangent, invEnvRot);
			++reservoir.M;
			for (int i = 0; i < RESERVOIR_SIZE; ++i)
				//除pdf是为了兼容非均匀采样
				if (updateReservoir(reservoir, i, tempPHat / pdf, sampleSeed, seed))
					pHat = tempPHat;
		}

		//注释掉这段代码会导致numCandidates=1的时候时域累积有扩散的黑点
		if (VISIBILITY_REUSE == 1)
		{
			uint sampleSeed = reservoir.samples[0].sampleSeed;
			rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, sampleEnvironmentMap(pdf, sampleSeed, normal, tangent, bitangent, invEnvRot), 100000);
			traceTransparency(rayQuery, seed);
			if (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
				reservoir.samples[0].sumWeights = 0.0;
		}

		if (TEMPORAL_REUSE_DI == 1 && currentFactor < 1.0)
		{
			Reservoir previousReservoir = readPreviousReservoir(previousPixel);
			previousReservoir.M = min(previousReservoir.M, ubo.temporalSamplesDI * reservoir.M);
			float previousPHat[RESERVOIR_SIZE];
			for (int i = 0; i < RESERVOIR_SIZE; ++i)
			{
				uint sampleSeed = previousReservoir.samples[i].sampleSeed;
				previousPHat[i] = evaluatePHat(pdf, sampleSeed, hitMaterial, view, normal, tangent, bitangent, invEnvRot);
			}
			if (combineReservoirs(reservoir, previousReservoir, previousPHat, seed))
				pHat = previousPHat[0];
			//weight用pHat[0]/pdf结果不对，即使numCandidates=1也不对
			//只有在用reservoir更新previousReservoir时weight能用pHat[0]/pdf
			//因为pHat[i]*other.samples[i].W*other.M包含了initialSample时计算的1/pdf
			//updateReservoir(reservoir, 0, pHat[0] / pdf, previousReservoir.samples[0].sampleSeed, seed);
			//updateReservoir(reservoir, 0, pHat[0] * previousReservoir.samples[0].W * previousReservoir.M, previousReservoir.samples[0].sampleSeed, seed);
			//reservoir.M += previousReservoir.M;
		}

		//加上这句RIS就不会过亮了，在if(rnd(seed)<replacePossibility)里给reservoir.samples[0].W赋值是错的
		for (int i = 0; i < RESERVOIR_SIZE; ++i)
			reservoir.samples[i].W = reservoir.samples[i].sumWeights / max(reservoir.M * pHat, 0.00001);
	}
	
	//放到ubo.ris==1里开关RIS有黑斑
	//必须放到空域复用之前
	writeReservoir(pixel, reservoir);

	if (RESTIR_DI == 1 && NEE == 1)
	{
		float pdf = 1 / (2 * M_PI);//1 / (2 * pi) * pi
		if (SPATIAL_REUSE_DI == 1)
		{
			vec2 texelSize = 1.0 / resolution;
			float radius = ubo.spatialReuseRadiusDI;
			vec3 R_sPosition = fragPos + 0.00001 * normal;
			vec2 neighborsUV[10];
			uint neighborsM[10];
			int numNeighbors = 0;
			int selected = -1;

			for (int s = 0; s < ubo.spatialSamplesDI; ++s)
			{
				vec2 offsetUV = randomChooseNeighbor(seed, radius) * texelSize;
				vec2 neighborUV = previousUV + offsetUV;

				// 相似度比较（可选）
				if (GEOMETRIC_SIMILARITY_DI == 1)
				{
					bool sim = calcGeoSimilarity(inUV, neighborUV);
					if (!sim)
						continue;
				}

				int neighborReservoirIndex = int(resolution.x * uint(resolution.y * neighborUV.y) + resolution.x * neighborUV.x);
				Reservoir neighborReservoir = readPreviousReservoir(neighborReservoirIndex);
				float neighborPHat[RESERVOIR_SIZE];
				uint sampleSeed = neighborReservoir.samples[0].sampleSeed;
				vec3 dir = sampleEnvironmentMap(pdf, sampleSeed, normal, tangent, bitangent, invEnvRot);
				neighborPHat[0] = evaluatePHat(dir, hitMaterial, view, normal, invEnvRot);
				if (neighborPHat[0] > 0.0)
				{
					rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, R_sPosition, 0.0, dir, 10000);
					rayQueryProceedEXT(rayQuery);
					traceTransparency(rayQuery, seed);
					if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
						continue;
						//neighborPHat=0很暗
						//neighborPHat[0] = 0.0;
				}
				if (combineReservoirs(reservoir, neighborReservoir, neighborPHat, seed))
				{
					pHat = neighborPHat[0];
					selected = numNeighbors;
				}
				neighborsUV[numNeighbors] = inUV + offsetUV;
				neighborsM[numNeighbors] = neighborReservoir.M;
				++numNeighbors;
			}
			if (BIAS_CORRECTION_DI == 1)
			{
				float selectedPHat;
				float sumPHat;
				if (SPATIAL_RIS_WITH_MIS_DI == 1)
				{
					selectedPHat = pHat;
					sumPHat = pHat * reservoirs[pixel].M;
				}
				else
					reservoir.M = reservoirs[pixel].M;
				vec3 neighborTangent;
				vec3 neighborBitangent;
				for (int i = 0; i < numNeighbors; ++i)
				{
					vec2 neighborUV = neighborsUV[i];
					vec3 neighborNormal = texture(samplerNormal, neighborUV).xyz;
					vec3 neighborPosition = texture(samplerPosition, neighborUV).xyz;
					HitMaterial neghborMaterial = readMaterial(neighborUV);
					uint sampleSeed = reservoir.samples[0].sampleSeed;
					computeDefaultBasis(neighborNormal, neighborTangent, neighborBitangent);
					vec3 neighborView = normalize(ubo.viewPos - neighborPosition);
					vec3 dir = sampleEnvironmentMap(pdf, sampleSeed, neighborNormal, neighborTangent, neighborBitangent, invEnvRot);
					float currentPHat = evaluatePHat(dir, neghborMaterial, neighborView, neighborNormal, invEnvRot);
					if (currentPHat > 0.0 && BIAS_CORRECTION_RAY_TRACED_DI == 1)
					{
						vec3 origin = neighborPosition + 0.00001 * neighborNormal;
						rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, dir, 10000);
						rayQueryProceedEXT(rayQuery);
						traceTransparency(rayQuery, seed);
						if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
							currentPHat = 0.0;
					}
					if (SPATIAL_RIS_WITH_MIS_DI == 1)
					{
						selectedPHat = selected == i ? currentPHat : selectedPHat;
						sumPHat += currentPHat * neighborsM[i];
					}
					else if (currentPHat > 0.0)
						reservoir.M += neighborsM[i];
				}
				if (SPATIAL_RIS_WITH_MIS_DI == 1)
					reservoir.samples[0].W = reservoir.samples[0].sumWeights * selectedPHat / max(sumPHat * pHat, 0.00001);
			}
			if (SPATIAL_RIS_WITH_MIS_DI == 0 || BIAS_CORRECTION_DI == 0)
				for (int i = 0; i < RESERVOIR_SIZE; ++i)
					reservoir.samples[i].W = reservoir.samples[i].sumWeights / max(reservoir.M * pHat, 0.00001);
		}
	}

	// Load the GMM statistics
	// --------------------------------------------------------------------
	GMM2D gmms;
	MultivariateGaussian2D gmm;
	vec4 pack0 = vec4(0.0);
	vec4 pack1 = vec4(0.0);
	if (SSPG == 1)
	{
		if (currentFactor < 1.0)
		{
			if (SGM == 1)
			{
				pack0 = gmmStatisticsPack0Prev[pixel];
				pack1 = gmmStatisticsPack1Prev[pixel];
			}
			else
				for (int i = 0; i < lobeCount; ++i)
				{
					gmms.sufficientStats0[i] = gmmStatisticsPack0Prev[lobeIndex + i];
					gmms.sufficientStats1[i] = gmmStatisticsPack1Prev[lobeIndex + i];
				}
		}
		else if (SGM == 0)
		{
			vec4 weights = vec4(0.1, 0.2, 0.3, 0.4);
			for (int i = 0; i < lobeCount; ++i)
			{
				vec2 uv = 1.0 + 1.0 * vec2(i / 2, i % 2);
				gmms.sufficientStats0[i] = vec4(uv, 0.01, 0.01);
				gmms.sufficientStats1[i] = vec4(0.0, 1.0, 0.0, weights[0]);
            }
        }

		if (SGM == 0)
			buildGMMs(gmms);
		else
		{
			uint epoch_count = uint(pack1.z);
			GMMStatictics GMMstat = unpackStatistics(pack0, pack1);
			gmm = createMultivariateGaussian2D(vec2(GMMstat.ex, GMMstat.ey), createCovariance(GMMstat));
		}
	}

	IndirectReservoir indirectReservoir = createIndirectReservoir();
	vec3 directRadiance = vec3(0.0);
	bool findSecondaryVertex = false;
	vec3 testColor = vec3(0.0);
	int sampleCount = ubo.sampleCount;
	PathVertex pathVertices[100];
	float primarySumWeights = 0.0;
	int primaryGridIndex = -1;
	float incidentRadiancePdf = 1 / (4 * M_PI);
	float solidAngle0 = 4 * M_PI / 1000 / 1000;
	for (int i = 0; i < sampleCount; ++i)
	//int i = 0;
	//用specialization constant没有提升
	//[[unroll]]
	//for (int i = 0; i < SAMPLECOUNT; ++i)
	{
		vec3 hitNormal = normal;
		vec3 hitPos = fragPos;
		vec3 hitView = view;
		vec3 throughput = vec3(1.0);
		//俄罗斯轮盘赌概率
		float russianRoulette = 0.5;
		for (int j = 0; j < ubo.numBounces; ++j)
		{
			if (ubo.russianRoulette == 1 && j > 3)
			{
    			// 计算当前路径的镜面反射率
				float reflectance = max(hitMaterial.baseColor.r, max(hitMaterial.baseColor.g, hitMaterial.baseColor.b));
				// 使用俄罗斯轮盘赌算法来决定是否终止路径
   				if (rnd(seed) > russianRoulette)
    				break;
   				// 根据路径长度和概率缩放路径的贡献
   				throughput /= russianRoulette;
    			russianRoulette *= reflectance;
			}
			origin = hitPos + 0.00001 * hitNormal;
			computeDefaultBasis(hitNormal, tangent, bitangent);

			int gridIndex = -1;
			if (PATH_GUIDING == 1 && SSPG == 0)
			{
				gridIndex = getGridIndex(hitPos);
				if (j == 0)
					primaryGridIndex = gridIndex;
			}
	
			if (SPOT_LIGHT == 1)
			{
				vec3 lightDir = normalize(ubo.spotLightPos - hitPos);
				float spotDist = distance(hitPos, ubo.spotLightPos);
				float attenuation = ubo.spotLightIntensity / spotDist / spotDist;
				vec3 spotBRDF = evaluateBRDF(hitView, hitNormal, lightDir, hitMaterial);
				if (dot(-lightDir, ubo.spotDir) > cos(0.5 * ubo.spotAngle * M_PI / 180.0))
				{
					rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, lightDir, spotDist);
					traceTransparency(rayQuery, seed);
					if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT)
					{
						if (j > 0)
							indirectReservoir.radiance += throughput * spotBRDF * attenuation * max(0.0, dot(lightDir, hitNormal));
						else
							directRadiance += throughput * spotBRDF * attenuation * max(0.0, dot(lightDir, hitNormal));
						if (PATH_GUIDING == 1)
						{
							if (gridIndex >= 0)
							{
								if (VXPG == 1)
									updateBoundingVoxel(hitPos, vec3(attenuation));
								else
									updateRadianceField(hitPos, lightDir, vec3(attenuation));
							}
						}
					}
				}
			}

			if (NEE == 1)
			{
				vec3 neeDir;
				float neePdf;
				if (ENVIRONMENT_MAP == 1)
				{
					//直接光照NEE帧率更高，因为可以用gl_RayFlagsTerminateOnFirstHitEXT
					//但间接光照更慢，因为还要额外求交点
					if (RESTIR_DI == 1 && j == 0)
					{
						neeDir = sampleEnvironmentMap(neePdf, reservoir.samples[i].sampleSeed, hitNormal, tangent, bitangent, invEnvRot);
						neePdf = 1.0 / reservoir.samples[i].W;
					}
					else
						neeDir = sampleEnvironmentMap(neePdf, seed, hitNormal, tangent, bitangent, invEnvRot);
					
					float neeNDotL = max(0.0, dot(neeDir, hitNormal));
					//判断neePdf能去除扩散黑点，判断neeNDotL能加速
					if (neePdf > 0.0 && neeNDotL > 0.0)
					{
						if (NEE_MIS == 1)
							neePdf += pdfBRDF(hitView, hitNormal, neeDir, hitMaterial);
						else
							//double counting，所以要除以2
							neePdf *= 2.0;

						vec3 neeBRDF = evaluateBRDF(hitView, hitNormal, neeDir, hitMaterial);
						
						if (ALPHA_TEST == 1)
						{
							rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, neeDir, 10000);
							traceTransparency(rayQuery, seed);
						}
						else
						{
							rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | rayFlags, 0xFF, origin, 0.0, neeDir, 10000);
							while (rayQueryProceedEXT(rayQuery));
						}

						if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT)
						{
							vec3 envMapRadiance = sampleEnvMap(ubo.envRot * neeDir);
							if (j > 0)
								indirectReservoir.radiance += neeBRDF * throughput * max(0.0, dot(neeDir, hitNormal)) * envMapRadiance / neePdf;
							else
								directRadiance += neeBRDF * throughput * max(0.0, dot(neeDir, hitNormal)) * envMapRadiance / neePdf;
							if (PATH_GUIDING == 1)
							{
								if (gridIndex >= 0)
								{
									if (VXPG == 1)
										updateBoundingVoxel(hitPos, envMapRadiance);
									else
										updateRadianceField(hitPos, neeDir, envMapRadiance);
								}
							}
						}
					}
				}
			}

			IncidentRadianceReservoir incidentRadianceReservoir;
			if (PATH_GUIDING == 1 && SSPG == 0 && CDF == 0)
			{
				if (gridIndex >= 0)
				{
					incidentRadianceReservoir = createIncidentRadianceReservoir();
					int count = 0;
					//比k = 0; k < INCIDENT_RADIANCE_MAP_SIZE * INCIDENT_RADIANCE_MAP_SIZE快
					for (int y = 0; y < INCIDENT_RADIANCE_MAP_SIZE; ++y)
						for (int x = 0; x < INCIDENT_RADIANCE_MAP_SIZE; ++x)
						{
							uint index = y * INCIDENT_RADIANCE_MAP_SIZE + x;
							//这个判断很慢，而且偏暗
							//如果不判断而是作为权重乘呢？
							bool front = dot(hitNormal, octahedronToUnitVector(2.0 * vec2(x, y) / INCIDENT_RADIANCE_MAP_SIZE - 1.0)) > 0.0;
							//判断incidentRadianceCount很慢(可能因为访问数据多)
							//判断incidentRadiance快很多，不判断要再快一点
							//if (incidentRadianceGridCells[gridIndex].incidentRadiance[index] > 0.0)
							{
								updateIncidentRadianceReservoir(incidentRadianceReservoir, incidentRadianceGridCells[gridIndex].incidentRadiance[index], index, seed);
								++count;
							}
							/*if (true)
							{
								updateIncidentRadianceReservoir(incidentRadianceReservoir, 1.0, uvec2(x, y), seed);
								++count;
							}*/
						}
					//用count会偏亮
					incidentRadianceReservoir.W = incidentRadianceReservoir.sumWeights / max(INCIDENT_RADIANCE_MAP_SIZE * INCIDENT_RADIANCE_MAP_SIZE * incidentRadianceReservoir.weight, 0.00001);
					//加不加没有区别
					//if (count == 0)
					//	sampleIncidentRadiance = false;
					if (j == 0)
						primarySumWeights = incidentRadianceReservoir.sumWeights;
				}
			}

			//每次反弹要用不同的noise，否则会有pattern，如果animateNoise则会抖动
			//但一个像素的所有样本noise不能变，否则就和完全随机一样了
			uv = vec2(rnd(seed), rnd(seed));
			float nDotL = 0.0;
			float multiplier = 1.0;
			float pdf = 0.0;
			vec3 direction;
			bool guiding = PATH_GUIDING == 1;
			if (SSPG == 1)
			{
				if (j > 0)
					guiding = false;
			}
			else if (VXPG == 0)
				guiding = guiding && gridIndex >= 0;
			//if (j == 0)
			//	guiding = false;
			if (guiding && rnd(seed) < ubo.probability)
			{
				if (SSPG == 1)
				{
					mat3 frame = createFrame(hitNormal);
					vec2 gmmSample;
					if (SGM == 1)
					{
						gmmSample = drawSample(gmm, vec2(rnd(seed), rnd(seed)));
						pdf = pdfGMM(gmm, gmmSample) / float(2.0 * k_pi);
					}
					else
					{
						gmmSample = drawSample(gmms, vec3(rnd(seed), rnd(seed), rnd(seed)));
						pdf = pdfGMMs(gmms, gmmSample) / float(2.0 * k_pi);
					}
					direction = to_world(frame, concentricDiskToUniformHemisphere(toConcentricMap(gmmSample)));
					if (pdf <= 0.0 || any(lessThan(gmmSample, vec2(0.0))) || any(greaterThan(gmmSample, vec2(1.0))))
						break;
				}
				else if (VXPG == 1)
				{
					// VXPG path guiding
					float voxelPdf;
					int selectedVoxel = sampleVoxel(hitPos, hitNormal, seed, voxelPdf);
					
					if (selectedVoxel >= 0 && voxelPdf > 0.0)
					{
						vec3 sampledPos, sampledNormal;
						HitMaterial sampledMaterial;
						
						// Try to sample within the voxel
						if (sampleIntraVoxel(selectedVoxel, seed, sampledPos, sampledNormal, sampledMaterial))
						{
							direction = normalize(sampledPos - hitPos);
							nDotL = max(0.0, dot(hitNormal, direction));
							
							if (nDotL > 0.0)
							{
								// Compute PDF: voxel selection × intra-voxel sampling
								vec3 aabbMin = vec3(
									uintBitsToFloat(boundingVoxels[selectedVoxel].aabbMinX),
									uintBitsToFloat(boundingVoxels[selectedVoxel].aabbMinY),
									uintBitsToFloat(boundingVoxels[selectedVoxel].aabbMinZ)
								);
								vec3 aabbMax = vec3(
									uintBitsToFloat(boundingVoxels[selectedVoxel].aabbMaxX),
									uintBitsToFloat(boundingVoxels[selectedVoxel].aabbMaxY),
									uintBitsToFloat(boundingVoxels[selectedVoxel].aabbMaxZ)
								);
								float aabbVolume = max(1e-6, (aabbMax.x - aabbMin.x) * (aabbMax.y - aabbMin.y) * (aabbMax.z - aabbMin.z));
								float intraVoxelPdf = 1.0 / aabbVolume; // Uniform sampling in AABB
								
								pdf = voxelPdf * intraVoxelPdf * ubo.probability;
							}
							else
							{
								// Fallback to BSDF sampling if direction is invalid
								direction = sampleBRDF(uv.x, uv.y, rnd(seed), hitView, hitNormal, hitMaterial);
								nDotL = max(0.0, dot(hitNormal, direction));
								pdf = pdfBRDF(hitView, hitNormal, direction, hitMaterial) * (1.0 - ubo.probability);
							}
						}
						else
						{
							// Fallback to BSDF sampling if no geometry found
							direction = sampleBRDF(uv.x, uv.y, rnd(seed), hitView, hitNormal, hitMaterial);
							nDotL = max(0.0, dot(hitNormal, direction));
							pdf = pdfBRDF(hitView, hitNormal, direction, hitMaterial) * (1.0 - ubo.probability);
						}
					}
					else
					{
						// Fallback to BSDF sampling
						direction = sampleBRDF(uv.x, uv.y, rnd(seed), hitView, hitNormal, hitMaterial);
						nDotL = max(0.0, dot(hitNormal, direction));
						pdf = pdfBRDF(hitView, hitNormal, direction, hitMaterial) * (1.0 - ubo.probability);
					}
					
					if (nDotL <= 0.0 || pdf <= 0.0)
						break;
					
					if (GUIDING_MIS == 1)
						pdf += pdfBRDF(hitView, hitNormal, direction, hitMaterial) * (1.0 - ubo.probability);
				}
				else
				{
					if (CDF == 1)
					{
						int left = 0;
						int right = int(INCIDENT_RADIANCE_MAP_SIZE * INCIDENT_RADIANCE_MAP_SIZE);
						float rndValue = rnd(seed);
						// incidentRadianceReservoir = createIncidentRadianceReservoir();
						while(left <= right){
							int mid = (right - left) / 2 + left;
							if(incidentRadianceGridCells[gridIndex].cdf[mid] < rndValue){
								left = mid + 1;
							}
							else{
								right = mid - 1;
							}
						} 
						incidentRadianceReservoir.index = left;
						/*for (int i = 0; i < INCIDENT_RADIANCE_MAP_SIZE * INCIDENT_RADIANCE_MAP_SIZE; ++i)
							if (incidentRadianceGridCells[gridIndex].cdf[i] > rndValue)
							{
								index = i;
								break;
							}*/
					}
					uvec2 selectedCoord;
					//selectedCoord.x = uint (mod(float(incidentRadianceReservoir.index), float(INCIDENT_RADIANCE_MAP_SIZE)));
					//selectedCoord.y = (incidentRadianceReservoir.index - x) / INCIDENT_RADIANCE_MAP_SIZE;
					//对于CDF在这里算selectedCoord比在else里算selectedCoord帧率高
					selectedCoord.y = incidentRadianceReservoir.index / INCIDENT_RADIANCE_MAP_SIZE;
					selectedCoord.x = incidentRadianceReservoir.index - selectedCoord.y * INCIDENT_RADIANCE_MAP_SIZE;
					vec2 octahedralCoord = 2.0 * (selectedCoord + uv) / INCIDENT_RADIANCE_MAP_SIZE - 1.0;
					float solidAngle = octahedralSolidAngle(octahedralCoord, 1.0 / 1000);
					direction = octahedronToUnitVector(octahedralCoord);
					if (CDF == 0)
						pdf = incidentRadiancePdf * ubo.probability / incidentRadianceReservoir.W * solidAngle0 / solidAngle;
					else
						pdf = incidentRadiancePdf * ubo.probability * incidentRadianceGridCells[gridIndex].incidentRadiance[incidentRadianceReservoir.index] * solidAngle0 / solidAngle;
				}
				nDotL = max(0.0, dot(hitNormal, direction));
				//这里加能提升帧数
				if (nDotL <= 0.0)
					break;
				if (GUIDING_MIS == 1)
					pdf += pdfBRDF(hitView, hitNormal, direction, hitMaterial) * (1.0 - ubo.probability);
				else
				{
					if (DEBUG == 1)
						pdf *= 2.0;
					if (DEBUG == 2)
						pdf /= ubo.probability;
				}
			}
			else
			{
				direction = sampleBRDF(uv.x, uv.y, rnd(seed), hitView, hitNormal, hitMaterial);
				nDotL = max(0.0, dot(hitNormal, direction));
				if (nDotL <= 0.0)
					break;
				pdf = pdfBRDF(hitView, hitNormal, direction, hitMaterial);
				//能略微提高帧数
				if (pdf <= 0.0)
					break;
				if (guiding)
				{
					pdf *= 1.0 - ubo.probability;
					if (GUIDING_MIS == 1)
					{
						if (gridIndex >= 0)
						{
							ivec2 coord = clamp(ivec2(floor(INCIDENT_RADIANCE_MAP_SIZE * (unitVectorToOctahedron(direction) * 0.5 + vec2(0.5)))), ivec2(0), ivec2(INCIDENT_RADIANCE_MAP_SIZE - 1));
							uint index = coord.y * INCIDENT_RADIANCE_MAP_SIZE + coord.x;
							if (incidentRadianceGridCells[gridIndex].incidentRadiance[index] > 0.0)
							{
								float solidAngle = octahedralSolidAngle(unitVectorToOctahedron(direction), 1.0 / 1000);
								if (CDF == 0)
								{
									float W = incidentRadianceReservoir.sumWeights / max(INCIDENT_RADIANCE_MAP_SIZE * INCIDENT_RADIANCE_MAP_SIZE * incidentRadianceGridCells[gridIndex].incidentRadiance[index], 0.00001);
									pdf += incidentRadiancePdf * ubo.probability / W * solidAngle0 / solidAngle;
								}
								else
									pdf += incidentRadiancePdf * ubo.probability * incidentRadianceGridCells[gridIndex].incidentRadiance[index] * solidAngle0 / solidAngle;
							}
						}
					}
					else
					{
						if (DEBUG == 0)
							nDotL = 0.0;
						if (DEBUG == 1)
							pdf *= 2.0;
						if (DEBUG == 2)
							pdf /= (1.0 - ubo.probability);
					}
					//pathVertices[j].valid = false;
				}
			}

			vec3 brdf = evaluateBRDF(hitView, hitNormal, direction, hitMaterial);
			
			if (PATH_GUIDING == 1 && SSPG == 0)
			{
				pathVertices[j].position = hitPos;
				pathVertices[j].direction = direction;
				pathVertices[j].throughput = nDotL * brdf;
				if (luminance(brdf) > ubo.clampValue * 0.05)
					pathVertices[j].throughput = vec3(0.0);
			}

			if (ALPHA_TEST == 1)
			{
				rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsNoneEXT, 0xFF, origin, 0.0, direction, 10000);
				// Start traversal: return false if traversal is complete
				traceTransparency(rayQuery, seed);
			}
			else
			{
				rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, origin, 0.0, direction, 10000);
				while (rayQueryProceedEXT(rayQuery));
			}

			if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
			{
				readHit(rayQuery, hitPos, hitNormal, hitMaterial, direction);

				hitView = -direction;

				if (j == 0)
				{
					indirectReservoir.position = hitPos;
					//应该在读取material之后，否则没考虑normal map
					indirectReservoir.normal = hitNormal;
					findSecondaryVertex = true;
				}
				else
					throughput *= nDotL * brdf / pdf;
			}
			else
			{
				//用continue会过亮
				if (ENVIRONMENT_MAP == 1)
				{
					vec3 envMapRadiance = sampleEnvMap(ubo.envRot * direction);
					if (PATH_GUIDING == 1 && SSPG == 0)
						if (gridIndex >= 0)
						{
							if (VXPG == 1)
							{
								// For VXPG, update bounding voxels with hit positions
								vec3 incidentRadiance = envMapRadiance;
								for (int k = j; k >= 0; --k)
								{
									updateBoundingVoxel(pathVertices[k].position, incidentRadiance);
									incidentRadiance *= pathVertices[k].throughput;
								}
							}
							else
							{
								vec3 incidentRadiance = envMapRadiance;
								for (int k = j; k >= 0; --k)
								{
									updateRadianceField(pathVertices[k].position, pathVertices[k].direction, incidentRadiance);
									incidentRadiance *= pathVertices[k].throughput;
								}
							}
						}
					if (NEE == 1)
					{
						if (NEE_MIS == 1)
						{
							if (RESTIR_DI == 1 && j == 0)
							{
								float pHat = evaluatePHat(direction, hitMaterial, hitView, hitNormal, invEnvRot);
								float W = reservoir.samples[i].sumWeights / max(reservoir.M * pHat, 0.00001);
								pdf += 1.0 / W;
							}
							else
								pdf += pdfEnvironmentMap(direction, hitNormal);
						}
						else
							pdf *= 2.0;
					}
					if (j > 0)
						indirectReservoir.radiance += throughput * nDotL * envMapRadiance * brdf / pdf;
					else
						directRadiance += throughput * nDotL * envMapRadiance * brdf / pdf;
				}
				break;
			}
		}
	}
	directRadiance /= float(sampleCount);
	indirectReservoir.radiance /= float(sampleCount);
	if (luminance(indirectReservoir.radiance) > ubo.clampValue)
		indirectReservoir.radiance = vec3(0.0);
	indirectReservoir.radiance = max(vec3(0.0), indirectReservoir.radiance);
	vec3 primaryBRDF = vec3(0.0);
	//不全部findSecondaryVertex是因为有nDotL<=0.0
	if (findSecondaryVertex)
	{
		vec3 dir = normalize(indirectReservoir.position - fragPos);
		primaryBRDF = max(0.0, dot(normal, dir)) * evaluateBRDF(view, normal, dir, primaryMaterial);
		if (SSPG == 1)
		{
			vpls[pixel] = vec4(indirectReservoir.position, luminance(directRadiance + indirectReservoir.radiance));
			if (SGM == 1)
			{
				gmmStatisticsPack0[pixel] = pack0;
				gmmStatisticsPack1[pixel] = pack1;
			}
			else
				for (int i = 0; i < lobeCount; ++i)
				{
					gmmStatisticsPack0[lobeIndex + i] = gmms.sufficientStats0[i];
					gmmStatisticsPack1[lobeIndex + i] = gmms.sufficientStats1[i];
				}
		}
	}
	//和以前的indirectReservoir.radiance相比，少除一个primaryHit采样的pdf
	//primaryBRDF有可能为0
	//不乘primaryBRDF会有firefly
	pHat = luminance(indirectReservoir.radiance * primaryBRDF);
	indirectReservoir.sumWeights += pHat;
	++indirectReservoir.M;
	if (RESTIR_GI == 1)
	{
		if (TEMPORAL_REUSE_GI == 1 && currentFactor < 1.0)
		{
			HitMaterial previousMaterial;
			vec4 specular = texture(samplerSpecular, previousUV);
			previousMaterial.baseColor = texture(samplerAlbedo, previousUV).rgb;
			previousMaterial.specular = 0.5;
			if (specular.a > 0.5)
			{
				previousMaterial.metallic = specular.r;
				previousMaterial.metallicRoughness = true;
			}
			else
			{
				previousMaterial.specularColor = specular.rgb;
				previousMaterial.metallicRoughness = false;
			}
			previousMaterial.roughness = texture(samplerMotion, previousUV).w;
			IndirectReservoir previousIndirectReservoir = readPreviousIndirectReservoir(previousPixel);
			previousIndirectReservoir.M = min(previousIndirectReservoir.M, ubo.temporalSamplesGI * indirectReservoir.M);
			vec3 dir = normalize(previousIndirectReservoir.position - fragPos);
			vec3 previousBRDF = max(0.0, dot(normal, dir)) * evaluateBRDF(view, normal, dir, primaryMaterial);
			float previousPHat = luminance(previousIndirectReservoir.radiance * previousBRDF);
			if (combineIndirectReservoirs(indirectReservoir, previousIndirectReservoir, previousPHat, seed))
				pHat = previousPHat;
		}
		indirectReservoir.W = indirectReservoir.sumWeights / max(indirectReservoir.M * pHat, 0.00001);
	}
	writeIndirectReservoir(pixel, indirectReservoir);
	if (RESTIR_GI == 1)
	{
		if (SPATIAL_REUSE_GI == 1)
		{
			vec2 texelSize = 1.0 / resolution;
			float radius = ubo.spatialReuseRadiusGI;
			vec2 neighborsUV[10];
			uint neighborsM[10];
			int numNeighbors = 0;
			int selected = -1;
			for (int s = 0; s < ubo.spatialSamplesGI; ++s)
			{
				vec2 offsetUV = randomChooseNeighbor(seed, radius) * texelSize;	//随机选取周围的点
				vec2 neighborUV = previousUV + offsetUV;

				if (GEOMETRIC_SIMILARITY_GI == 1)
				{
					bool sim = calcGeoSimilarity(inUV, neighborUV);	//相似度比较
					if (sim == false)
						continue;
				}

				int neighborReservoirIndex = int(resolution.x * uint(resolution.y * neighborUV.y) + resolution.x * neighborUV.x);
				IndirectReservoir neighborReservoir = readPreviousIndirectReservoir(neighborReservoirIndex);
				//计算雅可比行列式 得存坐标
				float jacobian = 1.0;
				vec3 neighborPos = texture(samplerPosition, neighborUV).xyz; //临近像素的世界坐标
				if (JACOBIAN == 1)
				{
					vec3 w1 = fragPos - neighborReservoir.position;
					vec3 w2 = neighborPos - neighborReservoir.position;
					float dist1 = dot(w1, w1);
					float dist2 = dot(w2, w2);
					float cos1 = clamp(dot(neighborReservoir.normal, normalize(w1)), 0.0, 1.0);
					float cos2 = clamp(dot(neighborReservoir.normal, normalize(w2)), 0.0, 1.0);
					jacobian = (cos1 * dist2) / (cos2 * dist1);
					if (!RAB_ValidateGISampleWithJacobian(jacobian))
						continue;
				}
				vec3 dir = normalize(neighborReservoir.position - fragPos);
				vec3 currentBRDF = max(0.0, dot(normal, dir)) * max(vec3(0.0), evaluateBRDF(view, normal, dir, primaryMaterial));
				//这里乘jacobian影响很小
				float neighborPHat = luminance(neighborReservoir.radiance * currentBRDF) * jacobian;
				if (neighborPHat > 0.0)
				{
					vec3 dir = normalize(neighborReservoir.position - fragPos);
					vec3 origin = fragPos + 0.00001 * normal;
					rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, dir, distance(origin, neighborReservoir.position) - 0.0001);
					rayQueryProceedEXT(rayQuery);
					traceTransparency(rayQuery, seed);
					if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
						//continue稍亮，neighborPHat=0稍暗
						continue;
						//neighborPHat = 0.0;
				}
				if (combineIndirectReservoirs(indirectReservoir, neighborReservoir, neighborPHat, seed))
				{
					pHat = neighborPHat;
					selected = numNeighbors;
				}
				neighborsUV[numNeighbors] = inUV + offsetUV;
				neighborsM[numNeighbors] = neighborReservoir.M;
				++numNeighbors;
			}
			if (BIAS_CORRECTION_GI == 1)
			{
				float selectedPHat;
				float sumPHat;
				if (SPATIAL_RIS_WITH_MIS_GI == 1)
				{
					selectedPHat = pHat;
					sumPHat = pHat * indirectReservoirs[pixel].M;
				}
				else
					indirectReservoir.M = indirectReservoirs[pixel].M;
				for (int i = 0; i < numNeighbors; ++i)
				{
					vec2 neighborUV = neighborsUV[i];
					vec3 neighborNormal = texture(samplerNormal, neighborUV).xyz;
					vec3 neighborPosition = texture(samplerPosition, neighborUV).xyz;
					vec3 dir = normalize(indirectReservoir.position - neighborPosition);
					vec3 neighborView = normalize(ubo.viewPos - neighborPosition);
					vec3 neighborBRDF = max(0.0, dot(neighborNormal, dir)) * evaluateBRDF(neighborView, neighborNormal, dir, readMaterial(neighborUV));
					//neighborBRDF会导致扩散黑块
					float currentPHat = luminance(indirectReservoir.radiance * neighborBRDF);
					if (currentPHat > 0.0 && BIAS_CORRECTION_RAY_TRACED_GI == 1)
					{
						vec3 origin = neighborPosition + 0.00001 * neighborNormal;
						rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, dir, distance(indirectReservoir.position, origin));
						rayQueryProceedEXT(rayQuery);
						traceTransparency(rayQuery, seed);
						if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
							currentPHat = 0.0;
					}
					if (SPATIAL_RIS_WITH_MIS_GI == 1)
					{
						//currentPHat会导致黑块
						selectedPHat = selected == i ? currentPHat : selectedPHat;
						sumPHat += currentPHat * neighborsM[i];
					}
					else if (currentPHat > 0.0)
						indirectReservoir.M += neighborsM[i];
				}
				if (SPATIAL_RIS_WITH_MIS_GI == 1)
					//这句在把reservoir改为storage buffer后会导致扩散黑快，selectedPHat有问题
					indirectReservoir.W = indirectReservoir.sumWeights * selectedPHat / max(sumPHat * pHat, 0.00001);
			}
			if (SPATIAL_RIS_WITH_MIS_GI == 0 || BIAS_CORRECTION_GI == 0)
				//SPATIAL_RIS_WITH_MIS_GI=0时第一个循环要用continue，否则偏暗。墙根稍有不同，关temporal resampling明显不同
				indirectReservoir.W = indirectReservoir.sumWeights / max(indirectReservoir.M * pHat, 0.00001);
		}
		indirectReservoir.radiance *= indirectReservoir.W;
	}
	directRadiance = max(vec3(0.0), directRadiance);
	//和直接用radiance相比BRDF用的是当前帧而不是之前帧的，但是因为后面除了albedo所以区别不大
	//空域复用应该有区别
	vec3 dir = normalize(indirectReservoir.position - fragPos);
	float pdf = pdfBRDF(view, normal, dir, primaryMaterial);
	vec3 primaryThroughput = max(0.0, dot(normal, dir)) * evaluateBRDF(view, normal, dir, primaryMaterial);
	//不能用<=0，为什么？
	if (pdf <= 1e-6)
		primaryThroughput = vec3(0.0);
	else
	{
		if (primaryGridIndex >= 0)
		{
			//非GUIDING_MIS不应该这样算
			pdf *= 1.0 - ubo.probability;
			ivec2 coord = clamp(ivec2(floor(INCIDENT_RADIANCE_MAP_SIZE * (unitVectorToOctahedron(dir) * 0.5 + vec2(0.5)))), ivec2(0), ivec2(INCIDENT_RADIANCE_MAP_SIZE - 1));
			uint index = coord.y * INCIDENT_RADIANCE_MAP_SIZE + coord.x;
			if (incidentRadianceGridCells[primaryGridIndex].incidentRadiance[index] > 0.0)
			{
				//会导致扩散黑块
				float solidAngle = octahedralSolidAngle(unitVectorToOctahedron(dir), 1.0 / 1000);
				if (isnan(solidAngle))
					solidAngle = 0.0;
				if (CDF == 0)
				{
					float W = primarySumWeights / max(INCIDENT_RADIANCE_MAP_SIZE * INCIDENT_RADIANCE_MAP_SIZE * incidentRadianceGridCells[primaryGridIndex].incidentRadiance[index], 0.00001);
					pdf += incidentRadiancePdf * ubo.probability / W * solidAngle0 / solidAngle;
				}
				else
					pdf += incidentRadiancePdf * ubo.probability * incidentRadianceGridCells[primaryGridIndex].incidentRadiance[index] * solidAngle0 / solidAngle;
			}
		}
		primaryThroughput /= pdf;
	}
	vec3 radiance = directRadiance + indirectReservoir.radiance * primaryThroughput;
	if (specular.a > 0.5)
		radiance /= max(vec3(0.01), albedo);
	else
		radiance /= max(vec3(0.01), albedo + specular.rgb);
	//radiance = testColor;
	if (ANIMATE_NOISE == 1 && (ubo.historyLength > 1 || ubo.reference == 1))
	{
		ivec2 screenSize = ivec2(resolution);
		Bilinear bilinearFilter = getBilinearFilter(previousUV, screenSize);
		vec4 bilinearWeights = getBilinearCustomWeights(bilinearFilter, vec4(1.0));
		ivec2 uv = ivec2(bilinearFilter.origin);
		uv = clamp(uv, ivec2(0), screenSize);
		vec4 z;
		z.x = texelFetch(samplerPreviousDepth, uv, 0).r;
		z.y = texelFetch(samplerPreviousDepth, uv + ivec2(1, 0), 0).r;
		z.z = texelFetch(samplerPreviousDepth, uv + ivec2(0, 1), 0).r;
		z.w = texelFetch(samplerPreviousDepth, uv + 1, 0).r;
		vec4 bilateralWeights = getBilateralWeight(z, vec4(texture(samplerPreviousDepth, inUV).x));
		vec4 w = bilinearWeights;// * bilateralWeights;
		vec3 s00 = texelFetch(samplerPreviousColor, uv, 0).rgb;
		vec3 s10 = texelFetch(samplerPreviousColor, uv + ivec2(1, 0), 0).rgb;
		vec3 s01 = texelFetch(samplerPreviousColor, uv + ivec2(0, 1), 0).rgb;
		vec3 s11 = texelFetch(samplerPreviousColor, uv + 1, 0).rgb;
		vec3 accumulatedRadiance = applyBilinearCustomWeights(s00, s10, s01, s11, w, false);
		//accumulatedRadiance = texture(samplerPreviousColor, previousUV).rgb;
		if (ubo.frameNumber != 0)
			outFragColor.rgb = mix(accumulatedRadiance, radiance, currentFactor);
		else
			outFragColor.rgb = radiance;
	}
	else
		outFragColor.rgb = radiance;//testColor;//

	/*if (DEBUG == 1)
	{
		outFragColor.rgb = vec3(0.0);
		ivec3 gridCell = ivec3((fragPos - gridBegin) / cellSize);
		if (gridCell.x >= 0 && gridCell.x < gridDim.x && gridCell.y >= 0 && gridCell.y < gridDim.y && gridCell.z >= 0 && gridCell.z < gridDim.z)
		{
			int gridIndex = gridCell.z * gridDim.x * gridDim.y + gridCell.y * gridDim.x + gridCell.x;
			ivec2 coord = clamp(ivec2(round(INCIDENT_RADIANCE_MAP_SIZE * (unitVectorToOctahedron(normal) * 0.5 + vec2(0.5)))), ivec2(0), ivec2(INCIDENT_RADIANCE_MAP_SIZE - 1));
			if (incidentRadianceGridCells[gridIndex].incidentRadianceCount[coord.y * INCIDENT_RADIANCE_MAP_SIZE + coord.x] > 0)
				outFragColor.rgb = vec3(incidentRadianceGridCells[gridIndex].incidentRadianceSum[coord.y * INCIDENT_RADIANCE_MAP_SIZE + coord.x] / incidentRadianceGridCells[gridIndex].incidentRadianceCount[coord.y * INCIDENT_RADIANCE_MAP_SIZE + coord.x]);
			//uint sum = 0;
			//for (int i = 0; i < INCIDENT_RADIANCE_MAP_SIZE * INCIDENT_RADIANCE_MAP_SIZE; ++i)
			//	sum += incidentRadianceGridCells[gridIndex].incidentRadianceCount[i];
			//outFragColor.rgb = 0.001 * vec3(sum);
		}
	}*/
}
