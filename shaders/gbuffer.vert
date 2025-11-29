#version 460

#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 3) in vec3 inNormal;
layout (location = 4) in vec4 inTangent;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 3) out vec3 outPos;
layout (location = 4) out vec4 outTangent;
layout (location = 5) out vec4 outPreviousPosition;
layout (location = 6) out vec4 outCurrentPosition;

layout (push_constant, scalar) uniform PushConstants
{
	int modelIndex;
} pushConstants;

layout (binding = 0, scalar) uniform UBO
{
	mat4 projection;
	mat4 model;
	mat4 view;
	mat4 previousProjection;
	mat4 previousView;
	float nearPlane;
	float farPlane;
	float alphaCutoff;
	vec3 probePos;
} ubo;

void main() 
{
	vec4 pos = inPos;
	//if (pushConstants.modelIndex == 1)
	//	pos.xyz += ubo.probePos;

	gl_Position = ubo.projection * ubo.view * ubo.model * pos;

	outCurrentPosition = gl_Position;
	outPreviousPosition = ubo.previousProjection * ubo.previousView * ubo.model * pos;

	outUV = inUV;

	// Vertex position in view space
	//outPos = vec3(ubo.view * ubo.model * pos);
	outPos = vec3(ubo.model * pos);

	// Normal in view space
	//mat3 normalMatrix = transpose(inverse(mat3(ubo.view * ubo.model)));
	//outNormal = normalMatrix * inNormal;
	outNormal = inNormal;
	if (BUMPED == 1)
		outTangent = inTangent;
}
