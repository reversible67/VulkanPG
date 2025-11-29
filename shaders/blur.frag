#version 450

#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
#include "reservoir.glsl"

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;
layout (location = 1) out float outDepth;
layout (location = 2) out uint outHistory;
layout (location = 3) out vec4 outColor;

layout (binding = 0) uniform sampler2D samplerColor;
layout (binding = 1) uniform sampler2D samplerPosition;
layout (binding = 2) uniform sampler2D samplerNormal;
layout (binding = 3) uniform usampler2D samplerHistory;
layout (binding = 5, scalar) readonly buffer Reservoirs { PackedReservoir reservoirs[]; };
layout (binding = 6, scalar) writeonly buffer PreviousReservoirs { PackedReservoir previousReservoirs[]; };
layout (binding = 7, scalar) readonly buffer IndirectReservoirs { PackedIndirectReservoir indirectReservoirs[]; };
layout (binding = 8, scalar) writeonly buffer PreviousIndirectReservoirs { PackedIndirectReservoir previousIndirectReservoirs[]; };
layout (binding = 9, scalar) buffer readonly GMMStatisticsPack0 { vec4 gmmStatisticsPack0[]; };
layout (binding = 10, scalar) buffer readonly GMMStatisticsPack1 { vec4 gmmStatisticsPack1[]; };
layout (binding = 11, scalar) buffer writeonly GMMStatisticsPack0Prev { vec4 gmmStatisticsPack0Prev[]; };
layout (binding = 12, scalar) buffer writeonly GMMStatisticsPack1Prev { vec4 gmmStatisticsPack1Prev[]; };

layout (binding = 4) uniform UBO
{
	int blur;
	int radius;
	float depthSharpness;
	float normalSharpness;
} ubo;

layout (push_constant, scalar) uniform PushConstants
{
	int modelIndex;
	vec3 gridBegin;
	ivec3 gridDim;
	float cellSize;
	int lobeCount;
};

float falloff = 0.0;

vec3 blurFunction(vec2 uv, float r, vec3 centerColor, float centerDepth, vec3 centerNormal, inout float sumWeights)
{
    vec3 color = texture(samplerColor, uv).rgb;
	float depth = texture(samplerPosition, uv).w;
	vec3 normal = texture(samplerNormal, uv).xyz;

    float depthDiff = centerDepth - depth;
	float normalDiff = 1.0 - max(0.0, dot(centerNormal, normal));
	float depthWeight = min(1.0, ubo.depthSharpness / (abs(depthDiff) + 0.0001));
	float normalWeight = pow(max(0.0, dot(centerNormal, normal)), ubo.normalSharpness);
	//float colorWeight = exp(-distance(centerColor, color));
    //float weight = exp(-normalDiff * normalDiff * ubo.normalSharpness - depthDiff * depthDiff * ubo.depthSharpness);
	float weight = exp2(-r * r * falloff) * depthWeight * normalWeight;
	//float weight = depthWeight * normalWeight;// * colorWeight;
	sumWeights += weight;

    return weight * color;
}

vec3 blurFunction(vec2 uv, float r, vec3 centerColor, vec3 centerPosition, vec3 centerNormal, inout float sumWeights)
{
    vec3 color = texture(samplerColor, uv).rgb;
	vec3 position = texture(samplerPosition, uv).xyz;
	vec3 normal = texture(samplerNormal, uv).xyz;

    float depthDiff = abs(dot(position - centerPosition, centerNormal));
	float normalDiff = 1.0 - max(0.0, dot(centerNormal, normal));
	float depthWeight = min(1.0, 0.1 * ubo.depthSharpness / (abs(depthDiff) + 0.0001));
	float normalWeight = pow(max(0.0, dot(centerNormal, normal)), ubo.normalSharpness);
    //float weight = exp(-normalDiff * normalDiff * ubo.normalSharpness - depthDiff * depthDiff * ubo.depthSharpness);
	float weight = depthWeight * normalWeight;
	sumWeights += weight;

    return weight * color;
}

vec3 blurFunction(vec2 uv, float r, inout float sumWeights)
{
    vec3 color = texture(samplerColor, uv).rgb;

    float weight = 1.0;
	sumWeights += weight;

    return weight * color;
}

void main()
{
	outDepth = texture(samplerPosition, inUV).w;
	outHistory = texture(samplerHistory, inUV).x;

	outColor = texture(samplerColor, inUV);

	uvec2 resolution = textureSize(samplerNormal, 0);
	int pixel = int(resolution.x * uint(resolution.y * inUV.y) + resolution.x * inUV.x);

	if (SSPG == 1)
	{
		if (SGM == 1)
		{
			gmmStatisticsPack0Prev[pixel] = gmmStatisticsPack0[pixel];
			gmmStatisticsPack1Prev[pixel] = gmmStatisticsPack1[pixel];
		}
		else
		{
			int lobeIndex = pixel * lobeCount;
			for (int i = 0; i < lobeCount; ++i)
			{
				gmmStatisticsPack0Prev[lobeIndex + i] = gmmStatisticsPack0[lobeIndex + i];
				gmmStatisticsPack1Prev[lobeIndex + i] = gmmStatisticsPack1[lobeIndex + i];
			}
		}
	}

	previousReservoirs[pixel] = reservoirs[pixel];
	previousIndirectReservoirs[pixel] = indirectReservoirs[pixel];

	vec3 centerColor = texture(samplerColor, inUV).rgb;
	vec3 centerNormal = texture(samplerNormal, inUV).xyz;
	vec3 centerPosition = texture(samplerPosition, inUV).xyz;

	if (ubo.blur == 0 || length(centerNormal) < 0.1)
		outFragColor.rgb = centerColor;
	else
	{
		float centerDepth = texture(samplerPosition, inUV).w;
		vec3 centerPosition = texture(samplerPosition, inUV).xyz;
		vec3 result = vec3(0.0);
		float sumWeights = 0.0;
		//float radius = min(ubo.normalSharpness * centerColor.a, ubo.radius);
		float radius = ubo.radius;
		vec2 texelSize = 1.0 / vec2(textureSize(samplerColor, 0));
		float sigma = radius * 0.5;
		falloff = 1.0 / (2.0 * sigma * sigma);
		for (float x = -radius; x <= radius; x++)
		{
			float y = 0.0;
			for (float y = -radius; y <= radius; y++)
			{
				vec2 offset = vec2(x, y);
				//result += blurFunction(inUV + offset * texelSize, length(offset), sumWeights);
				result += blurFunction(inUV + offset * texelSize, length(offset), centerColor, centerDepth, centerNormal, sumWeights);
			}
		}
		outFragColor.rgb = result / sumWeights;
	}
}
