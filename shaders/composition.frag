#version 460

#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
#include "reservoir.glsl"

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inDir;

layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform sampler2D samplerPosition;
layout (binding = 1) uniform sampler2D samplerNormal;
layout (binding = 2) uniform sampler2D samplerAlbedo;
layout (binding = 3) uniform sampler2D samplerSpecular;
layout (binding = 4) uniform sampler2D samplerIndirect;
layout (binding = 5) uniform sampler2D samplerIndirectBlur;
layout (binding = 6) uniform sampler2D samplerEnvMap;
layout (binding = 7) uniform sampler2D samplerMotion;
layout (binding = 8) uniform usampler2D samplerHistory;
layout (binding = 11, scalar) readonly buffer Reservoirs { PackedReservoir reservoirs[]; };
layout (binding = 12, scalar) readonly buffer IndirectReservoirs { PackedIndirectReservoir indirectReservoirs[]; };
layout (binding = 13, scalar) buffer readonly VPLs { vec4 vpls[]; };
layout (binding = 14, scalar) buffer readonly GMMStatisticsPack0 { vec4 gmmStatisticsPack0[]; };
layout (binding = 15, scalar) buffer readonly GMMStatisticsPack1 { vec4 gmmStatisticsPack1[]; };
layout (binding = 16, scalar) buffer readonly GMMStatisticsPack0Prev { vec4 gmmStatisticsPack0Prev[]; };
layout (binding = 17, scalar) buffer readonly GMMStatisticsPack1Prev { vec4 gmmStatisticsPack1Prev[]; };
layout (binding = 18, scalar) buffer writeonly Screen { vec3 screen[]; };

layout (binding = 9) uniform UBOBlur
{
	int blur;
	int radius;
	float depthSharpness;
	float normalSharpness;
} uboBlur;

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
	int screenshot;
} ubo;

vec3 toneMappingACES(vec3 color, float adaptedLum)
{
	const float A = 2.51;
	const float B = 0.03;
	const float C = 2.43;
	const float D = 0.59;
	const float E = 0.14;

	color *= adaptedLum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
}

float falloff = 0.0;

vec4 blurFunction(vec2 uv, float r, float centerDepth, vec3 centerNormal, inout float sumWeights)
{
    vec4 color = texture(samplerIndirectBlur, uv);
	float depth = texture(samplerPosition, uv).w;
	vec3 normal = texture(samplerNormal, uv).xyz;

    float depthDiff = centerDepth - depth;
	float depthWeight = min(1.0, uboBlur.depthSharpness / (abs(depthDiff) + 0.0001));
	float normalWeight = pow(max(0.0, dot(centerNormal, normal)), uboBlur.normalSharpness);
	//float weight = exp2(-r * r * falloff) * depthWeight * normalWeight;
	float weight = depthWeight * normalWeight;
	sumWeights += weight;

    return weight * color;
}

void main()
{
	vec3 fragPos = texture(samplerPosition, inUV).xyz;
	vec3 normal = texture(samplerNormal, inUV).xyz;
	vec4 albedo = texture(samplerAlbedo, inUV);

	vec4 indirectColor = texture(samplerIndirectBlur, inUV);
	if (uboBlur.blur == 1 && length(normal) > 0.1)
	{
		float centerDepth = texture(samplerPosition, inUV).w;
		vec4 result = vec4(0.0);
		float sumWeights = 0;
		float radius = uboBlur.radius;
		vec2 texelSize = 1.0 / vec2(textureSize(samplerIndirectBlur, 0));
		float sigma = radius * 0.5;
		falloff = 1.0 / (2.0 * sigma * sigma);
		float x = 0.0f;
		for (float y = -radius; y <= radius; y++)
		{
			vec2 offset = vec2(x, y);
			result += blurFunction(inUV + offset * texelSize, length(offset), centerDepth, normal, sumWeights);
		}
		indirectColor = result / sumWeights;
	}

	uvec2 size = textureSize(samplerNormal, 0);
	int pixel = int(size.x * uint(size.y * inUV.y) + size.x * inUV.x);

	// Debug display
	if (ubo.displayDebugTarget > 0)
	{
		switch (ubo.displayDebugTarget)
		{
			case 1:
				outFragColor.rgb = fragPos;
				break;
			case 2:
				outFragColor.rgb = normal * 0.5 + 0.5;
				break;
			case 3:
				outFragColor.rgb = pow(albedo.rgb, vec3(1.0 / 2.2));
				break;
			case 4:
				//outFragColor.rgb = 0.00001 * vec3(float(floatBitsToUint(texture(samplerReservoir, inUV).r)));
				//outFragColor.rgb = 0.00001 * texture(samplerReservoir, inUV).rrr;
				outFragColor.rgb = 0.0000000001 * vec3(reservoirs[pixel].sampleSeed);
				//outFragColor.rgb = texture(samplerReservoirs[1], inUV).rgb;
				break;
			case 5:
				outFragColor.rgb = texture(samplerPosition, inUV).www * 0.03;
				break;
			case 6:
				//outFragColor.rgb = vec3(inUV, 0.0);
				//outFragColor.rgb = vec3(texture(samplerSpecular, inUV).a, texture(samplerMotion, inUV).w, 0.0);
				outFragColor.rgb = vec3(texture(samplerMotion, inUV).w);
				break;
			case 7:
				//0.4(exposure)¡Á0.5(albedo)
				outFragColor.rgb = texture(samplerIndirect, inUV).rgb;
				break;
			case 8:
				outFragColor.rgb = texture(samplerIndirectBlur, inUV).rgb;
				break;
			case 9:
				outFragColor.rgb = indirectReservoirs[pixel].radiance;
				break;
			case 10:
				outFragColor.rgb = vec3(reservoirs[pixel].W);
				//outFragColor.rgb = vec3(indirectReservoirs[pixel].W);
				break;
			case 11:
				//outFragColor.rgb = vec3(0.01 * reservoirs[pixel].M);
				outFragColor.rgb = vec3(0.01 * indirectReservoirs[pixel].M);
				break;
			case 12:
				//outFragColor.rgb = texture(samplerIndirect, inUV).aaa;
				outFragColor.rgb = texture(samplerSpecular, inUV).rgb;
				break;
			case 13:
				outFragColor.rgb = vec3(100.0 * length(texture(samplerMotion, inUV).xy));
				break;
			case 14:
				outFragColor.rgb = texture(samplerHistory, inUV).xxx / 40.0;
				break;
			case 15:
				if (inUV.y < 0.5)
					outFragColor.rgb = texture(samplerIndirect, inUV).aaa * 0.03;
				else
					outFragColor.rgb = texture(samplerPosition, inUV).www * 0.03;
				break;
			case 16:
				outFragColor.rgb = vpls[pixel].xyz;
				break;
			case 17:
				outFragColor.rgb = vpls[pixel].aaa;
				break;
			case 18:
				outFragColor.rgb = gmmStatisticsPack0Prev[pixel].xyz;
				break;
		}
		outFragColor.a = 1.0;
		return;
	}
	vec3 outColor;
	if (length(normal) < 0.1)
	{
		if (ENVIRONMENT_MAP == 1)
		{
			vec3 dir = ubo.envRot * normalize(inDir);
			outColor = dir;
			outColor = textureLod(samplerEnvMap, sphere2UV(dir), 0).rgb;
		}
		else
			outColor = vec3(0.0);
	}
	else
	{
		outColor = texture(samplerIndirectBlur, inUV).rgb;
		if (TEXTURED == 1)
		{
			vec4 specular = texture(samplerSpecular, inUV);
			if (specular.a > 0.5)
				outColor *= max(vec3(0.01), albedo.rgb);
			else
				outColor *= max(vec3(0.01), albedo.rgb + specular.rgb);
		}
	}
	if (ubo.screenshot == 1)
		screen[pixel] = outColor.bgr * pow(ubo.exposure, 2.2);
	outColor = pow(outColor, vec3(1.0 / 2.2)) * ubo.exposure;
	if (TONE_MAPPING == 1)
		outFragColor.rgb = vec3(1.0) - exp(-outColor * ubo.exposure);//toneMappingACES(max(vec3(0.0), outColor), ubo.exposure);
	else
		outFragColor.rgb = outColor;
}
