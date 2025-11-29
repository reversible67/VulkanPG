#version 460

#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 3) in vec3 inPos;
layout (location = 4) in vec4 inTangent;
layout (location = 5) in vec4 inPreviousPosition;
layout (location = 6) in vec4 inCurrentPosition;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;
layout (location = 3) out vec4 outSpecular;
layout (location = 4) out vec4 outMotion;

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
	vec3 viewPos;
} ubo;

layout (push_constant, scalar) uniform PushConstants
{
	int modelIndex;
} pushConstants;

layout (buffer_reference, scalar) buffer Materials { Material m[]; }; // Array of all materials on an object
layout (buffer_reference, scalar) buffer MaterialIndices { int i[]; }; // Material ID for each triangle
layout (binding = 1, scalar) buffer readonly SceneDescs { SceneDesc sceneDescs[]; } sceneDescs;
layout (binding = 2) uniform sampler2D samplerImages[];

float linearDepth(float depth)
{
	float z = depth * 2.0f - 1.0f;
	return (2.0f * ubo.nearPlane * ubo.farPlane) / (ubo.farPlane + ubo.nearPlane - z * (ubo.farPlane - ubo.nearPlane));	
}

void main()
{
	SceneDesc sceneDesc = sceneDescs.sceneDescs[pushConstants.modelIndex];
	MaterialIndices materialIndices  = MaterialIndices(sceneDesc.materialIndexAddress);
	Materials materials = Materials(sceneDesc.materialAddress);
	int materialIndex = materialIndices.i[gl_PrimitiveID];

	//linearDepth(gl_FragCoord.z)值比inCurrentPosition.w小
	outPosition = vec4(inPos, inCurrentPosition.w);
	outNormal = vec4(normalize(inNormal), 1.0);
	vec3 view = normalize(ubo.viewPos - inPos);
	if (dot(view, outNormal.xyz) < 0.0)
		outNormal.xyz = -outNormal.xyz;
	outAlbedo = vec4(1.0);
	outSpecular = vec4(0.0, 0.0, 0.0, 1.0);
	vec2 metallicRoughness = vec2(0.0, 1.0);
	vec4 specularGlossiness = vec4(0.0);
	if (materialIndex >= 0)
	{
		Material material = materials.m[materialIndex];
		if (BUMPED == 1 && material.normalImage >= 0)
		{
			uint normalImage = material.normalImage + sceneDesc.imageOffset;
			//有些物体切线为0或没有切线，放applyNormalMap内部无效
			if (length(inTangent.xyz) > 0.9)
				outNormal.xyz = applyNormalMap(outNormal.xyz, inTangent, texture(samplerImages[nonuniformEXT(normalImage)], inUV).xyz);
			//outAlbedo.rgb = normalize(inTangent.xyz);
		}
		vec4 baseColor = material.baseColor;
		if (material.baseColorImage >= 0)
		{
			uint baseColorImage = material.baseColorImage + sceneDesc.imageOffset;
			vec4 baseColorTexel = texture(samplerImages[nonuniformEXT(baseColorImage)], inUV);
			if (GAMMA_CORRECTION == 1)
				baseColorTexel.rgb = pow(baseColorTexel.rgb, vec3(2.2));
			baseColor *= baseColorTexel;
		}
		//baseColor = baseColor.aaaa;
		if (ALPHA_TEST == 1 && (material.alphaMode == 1 || material.alphaMode == 2))
			if (baseColor.a < ubo.alphaCutoff)
				discard;
		//if (material.alphaMode == 2)
		//{
		//	uint seed = tea(uint(1280 * gl_FragCoord.y + gl_FragCoord.x), 0);
		//	if (baseColor.a < rnd(seed))
		//		discard;
		//}
		//if (material.transmission > 0.001)
		//	discard;
		/*if (material.alphaMode == 2 || material.transmission > 0.001)
		{
			uint seed = tea(uint(1280 * gl_FragCoord.y + gl_FragCoord.x), 0);
			float blendFactor = baseColor.a;
			if (material.transmission > 0.001)
				blendFactor = 1.0 - material.transmission;
			if (blendFactor < rnd(seed))
				discard;
		}*/
		outAlbedo = baseColor;
		if (material.metallicRoughness)
			metallicRoughness = vec2(material.metallic, material.roughness);
		else
			specularGlossiness = vec4(material.specularFactor, material.roughness);
		if (material.metallicRoughnessImage >= 0)
		{
			uint metallicRoughnessImage = material.metallicRoughnessImage + sceneDesc.imageOffset;
			if (material.metallicRoughness)
				metallicRoughness *= texture(samplerImages[nonuniformEXT(metallicRoughnessImage)], inUV).bg;
			else
			{
				vec4 specularGlossinessTexel = texture(samplerImages[nonuniformEXT(metallicRoughnessImage)], inUV);
				if (GAMMA_CORRECTION == 1)
					specularGlossinessTexel.rgb = pow(specularGlossinessTexel.rgb, vec3(2.2));
				specularGlossiness.rgb *= specularGlossinessTexel.rgb;
				specularGlossiness.a *= specularGlossinessTexel.a;
			}
		}

		if (material.metallicRoughness)
			outSpecular = vec4(metallicRoughness.r, 0.0, 0.0, 1.0);
		else
		{
			metallicRoughness.g = 1.0 - specularGlossiness.a * specularGlossiness.a;
			outSpecular = vec4(specularGlossiness.rgb, 0.0);
			//if (ubo.debug == 1)
			//	specularGlossiness = decodeFloat2RGBA(metallicRoughness.r);
		}
		//if (ubo.debug == 1)
		//	outAlbedo.rgb = specularGlossiness.rgb;
	}

	outMotion.xy = inCurrentPosition.xy / inCurrentPosition.w - inPreviousPosition.xy / inPreviousPosition.w;
	outMotion.y = -outMotion.y;
	outMotion.z = inCurrentPosition.w - inPreviousPosition.w;
	outMotion.w = metallicRoughness.g;
}