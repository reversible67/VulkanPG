/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#define NRD_BILATERAL_WEIGHT_VIEWZ_SENSITIVITY 0.1
#define NRD_BILATERAL_WEIGHT_CUTOFF 0.03
#define M_PI 3.1415926535897932384626433832795f

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_ray_flags_primitive_culling : require

layout (constant_id = 0) const int TEXTURED = 1;
layout (constant_id = 1) const int BUMPED = 1;
layout (constant_id = 2) const int ORTHOGONALIZE = 1;
layout (constant_id = 3) const int GAMMA_CORRECTION = 1;
layout (constant_id = 4) const int ALPHA_TEST = 1;
layout (constant_id = 5) const int TONE_MAPPING = 0;
layout (constant_id = 6) const int ANIMATE_NOISE = 1;
layout (constant_id = 7) const int NEE = 0;
layout (constant_id = 8) const int RESTIR_DI = 0;
layout (constant_id = 9) const int ENV_MAP_IS = 0;
layout (constant_id = 10) const int VISIBILITY_REUSE = 1;
layout (constant_id = 11) const int TEMPORAL_REUSE_DI = 1;
layout (constant_id = 12) const int TEMPORAL_REUSE_GI = 1;
layout (constant_id = 13) const int SPATIAL_REUSE_DI = 1;
layout (constant_id = 14) const int SPATIAL_REUSE_GI = 1;
layout (constant_id = 15) const int RESTIR_GI = 0;
layout (constant_id = 16) const int DEBUG = 0;
layout (constant_id = 17) const int SPOT_LIGHT = 0;
layout (constant_id = 18) const int ENVIRONMENT_MAP = 1;
layout (constant_id = 19) const int BIAS_CORRECTION_DI = 1;
layout (constant_id = 20) const int BIAS_CORRECTION_GI = 1;
layout (constant_id = 21) const int GEOMETRIC_SIMILARITY_DI = 1;
layout (constant_id = 22) const int GEOMETRIC_SIMILARITY_GI = 1;
layout (constant_id = 23) const int JACOBIAN = 1;
layout (constant_id = 24) const uint INCIDENT_RADIANCE_MAP_SIZE = 1;
layout (constant_id = 25) const int PATH_GUIDING = 0;
layout (constant_id = 26) const int NEE_MIS = 0;
layout (constant_id = 27) const int GUIDING_MIS = 0;
layout (constant_id = 28) const int HASHING = 0;
layout (constant_id = 29) const int CDF = 0;
layout (constant_id = 30) const int SSPG = 0;
layout (constant_id = 31) const int SGM = 0;
layout (constant_id = 32) const int VXPG = 0;

struct Vertex
{
	vec3 pos;
	vec3 normal;
	vec2 uv;
	vec4 color;
	vec4 joint0;
	vec4 weight0;
	vec4 tangent;
};

struct Material
{
	vec4 baseColor;
	float metallic;
	float roughness;
	int baseColorImage;
	int metallicRoughnessImage;
	int normalImage;
	int emissiveImage;
	int alphaMode;
	float transmission;
	bool metallicRoughness;
	vec3 specularFactor;
};

struct HitMaterial
{
    vec3 baseColor;
    float metallic;
    float specular;
    float roughness;
	bool metallicRoughness;
	vec3 specularColor;
};

struct SceneDesc
{
	int modelIndex;
	int imageOffset;
	uint64_t vertexAddress;
	uint64_t indexAddress;
	uint64_t materialAddress;
	uint64_t materialIndexAddress;
	uint64_t firstPrimitiveAddress;
};

struct IncidentRadianceGridCell
{
	float incidentRadianceSum[64];
	uint incidentRadianceCount[64];
	float incidentRadiance[64];
	float cdf[64];
};

struct BoundingVoxel
{
	uint aabbMinX;
	uint aabbMinY;
	uint aabbMinZ;
	float totalIrradiance;
	uint aabbMaxX;
	uint aabbMaxY;
	uint aabbMaxZ;
	uint sampleCount;
};

struct GeometryInfo
{
	int firstPrimitive;
};

vec3 to_world(mat3 frame, vec3 v)
{
	return v[0] * frame[0] + v[1] * frame[1] + v[2] * frame[2];
}

vec3 to_local(mat3 frame, vec3 v)
{
	return vec3(dot(v, frame[0]), dot(v, frame[1]), dot(v, frame[2]));
}

mat3 createFrame(vec3 n)
{
	if (n[2] < float(-1 + 1e-6))
		return mat3(vec3(0, -1, 0), vec3(-1, 0, 0), n);
	else
	{
		float a = 1 / (1 + n[2]);
		float b = -n[0] * n[1] * a;
		return mat3(vec3(1 - n[0] * n[0] * a, b, -n[0]), vec3(b, 1 - n[1] * n[1] * a, -n[1]), n);
	}
}

#define C_Stack_Max 3.402823466e+38f
uint CompressUnitVec(vec3 nv)
{
  // map to octahedron and then flatten to 2D (see 'Octahedron Environment Maps' by Engelhardt & Dachsbacher)
  if((nv.x < C_Stack_Max) && !isinf(nv.x))
  {
    const float d = 32767.0f / (abs(nv.x) + abs(nv.y) + abs(nv.z));
    int         x = int(roundEven(nv.x * d));
    int         y = int(roundEven(nv.y * d));
    if(nv.z < 0.0f)
    {
      const int maskx = x >> 31;
      const int masky = y >> 31;
      const int tmp   = 32767 + maskx + masky;
      const int tmpx  = x;
      x               = (tmp - (y ^ masky)) ^ maskx;
      y               = (tmp - (tmpx ^ maskx)) ^ masky;
    }
    uint packed = (uint(y + 32767) << 16) | uint(x + 32767);
    if(packed == ~0u)
      return ~0x1u;
    return packed;
  }
  else
  {
    return ~0u;
  }
}

float ShortToFloatM11(const int v)  // linearly maps a short 32767-32768 to a float -1-+1 //!! opt.?
{
  return (v >= 0) ? (uintBitsToFloat(0x3F800000u | (uint(v) << 8)) - 1.0f) :
                    (uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-v) << 8)) + 1.0f);
}
vec3 DecompressUnitVec(uint packed)
{
  if(packed != ~0u)  // sanity check, not needed as isvalid_unit_vec is called earlier
  {
    int       x     = int(packed & 0xFFFFu) - 32767;
    int       y     = int(packed >> 16) - 32767;
    const int maskx = x >> 31;
    const int masky = y >> 31;
    const int tmp0  = 32767 + maskx + masky;
    const int ymask = y ^ masky;
    const int tmp1  = tmp0 - (x ^ maskx);
    const int z     = tmp1 - ymask;
    float     zf;
    if(z < 0)
    {
      x  = (tmp0 - ymask) ^ maskx;
      y  = tmp1 ^ masky;
      zf = uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-z) << 8)) + 1.0f;
    }
    else
    {
      zf = uintBitsToFloat(0x3F800000u | (uint(z) << 8)) - 1.0f;
    }
    return normalize(vec3(ShortToFloatM11(x), ShortToFloatM11(y), zf));
  }
  else
  {
    return vec3(C_Stack_Max);
  }
}


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//
vec3 offsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}


//////////////////////////// AO //////////////////////////////////////
#define EPS 0.05

void computeDefaultBasis(const vec3 normal, out vec3 x, out vec3 y)
{
  // ZAP's default coordinate system for compatibility
  vec3        z  = normal;
  const float yz = -z.y * z.z;
  y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

  x = cross(y, z);
}

//-------------------------------------------------------------------------------------------------
// Random
//-------------------------------------------------------------------------------------------------


// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

uvec2 pcg2d(uvec2 v)
{
  v = v * 1664525u + 1013904223u;

  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;

  v = v ^ (v >> 16u);

  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;

  v = v ^ (v >> 16u);

  return v;
}

// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float in [0, 1) given the previous RNG state
float rnd(inout uint seed)
{
  return (float(lcg(seed)) / float(0x01000000));
}

vec3 hemisphereSample_uniform(vec2 uv)
{
	float phi = uv.y * 2.0 * M_PI;
	float cosTheta = 1.0 - uv.x;
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

vec3 hemisphereSample_cos(vec2 uv)
{
	float phi = uv.y * 2.0 * M_PI;
	float cosTheta = sqrt(1.0 - uv.x);
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// 将向量 v 投影到 N 的法向半球
vec3 toNormalHemisphere(vec3 v, vec3 N)
{
    vec3 helper = vec3(1, 0, 0);
    if(abs(N.x)>0.999) helper = vec3(0, 0, 1);
    vec3 tangent = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

vec2 sphere2UV(vec3 dir)
{
    return vec2(atan(dir.z, dir.x), asin(-dir.y)) / vec2(2.0 * M_PI, M_PI) + 0.5;
}


struct Bilinear { vec2 origin; vec2 weights; };

Bilinear getBilinearFilter( vec2 uv, vec2 texSize )
{
    vec2 t = uv * texSize - 0.5;

    Bilinear result;
    result.origin = floor( t );
    result.weights = t - result.origin;

    return result;
}

vec4 getBilinearCustomWeights( Bilinear f, vec4 customWeights )
{
	vec2 oneMinusWeights = 1.0 - f.weights;

	vec4 weights = customWeights;
	weights.x *= oneMinusWeights.x * oneMinusWeights.y;
	weights.y *= f.weights.x * oneMinusWeights.y;
	weights.z *= oneMinusWeights.x * f.weights.y;
	weights.w *= f.weights.x * f.weights.y;

	return weights;
}

vec3 applyBilinearCustomWeights( vec3 s00, vec3 s10, vec3 s01, vec3 s11, vec4 w, bool normalize )
{
	vec3 r = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
	return r * ( normalize ? 1.0 / ( dot( w, vec4(1.0) ) ) : 1.0 );
}

vec4 getBilateralWeight( vec4 z, vec4 zc )
{
    z = abs( z - zc ) * 1.0 / ( min( abs( z ), abs( zc ) ) + 0.001 ); \
    z = 1.0 / ( 1.0 + NRD_BILATERAL_WEIGHT_VIEWZ_SENSITIVITY * z ) * step(z, vec4(NRD_BILATERAL_WEIGHT_CUTOFF));
	return z;
}


float SchlickFresnel(float u)
{
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

float GTR1(float NdotH, float a)
{
    if (a >= 1) return 1/M_PI;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (M_PI*log(a2)*t);
}

float GTR2(float NdotH, float a)
{
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return a2 / (M_PI * t*t);
}

vec3 sampleGTR1(float xi_1, float xi_2, vec3 V, vec3 N, float alpha) {
    
    float phi_h = 2.0 * M_PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0-pow(alpha*alpha, 1.0-xi_2))/(1.0-alpha*alpha));
    float sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));

    // 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
    vec3 H = vec3(sin_theta_h*cos_phi_h, sin_theta_h*sin_phi_h, cos_theta_h);
    H = toNormalHemisphere(H, N);   // 投影到真正的法向半球

    // 根据 "微法线" 计算反射光方向
    vec3 L = reflect(-V, H);

    return L;
}

vec3 sampleGTR2(float xi_1, float xi_2, vec3 V, vec3 N, float alpha)
{
    float phi_h = 2.0 * M_PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0-xi_2)/(1.0+(alpha*alpha-1.0)*xi_2));
    float sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));

    // 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
    vec3 H = vec3(sin_theta_h*cos_phi_h, sin_theta_h*sin_phi_h, cos_theta_h);
    H = toNormalHemisphere(H, N);   // 投影到真正的法向半球

    // 根据 "微法线" 计算反射光方向
    vec3 L = reflect(-V, H);

    return L;
}

float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
	//加epsilon能避免neighborBRDF导致的扩散黑块
    return 1.0 / (NdotV + sqrt(a + b - a * b) + 0.00001);
}

float sqr(float x)
{
    return x*x; 
}

vec3 evaluateBRDF(vec3 V, vec3 N, vec3 L, in HitMaterial material)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    // 各种颜色
    vec3 Cspec0 = material.specularColor;
	if (material.metallicRoughness)
		Cspec0 = mix(0.08 * vec3(material.specular), material.baseColor, material.metallic); // 0° 镜面反射颜色

    float Fd90 = 0.5 + 2.0 * LdotH * LdotH * material.roughness;
    float FL = SchlickFresnel(NdotL);
    float FV = SchlickFresnel(NdotV);
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    // 镜面反射 -- 各向同性
    float alpha = max(0.001, sqr(material.roughness));
    float Ds = GTR2(NdotH, alpha);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0, vec3(1), FH);
    float Gs = smithG_GGX(NdotL, material.roughness);
	//如果isinf置0也可以避免neighborBRDF导致的扩散黑块
	//if (isinf(Gs))
	//	Gs = 0.0;
	//else
    Gs *= smithG_GGX(NdotV, material.roughness);

    vec3 diffuse = material.baseColor * Fd / M_PI;
	if (material.metallicRoughness)
		diffuse *= (1.0 - material.metallic);
    vec3 specular = Gs * Fs * Ds;

	if (NdotL < 0)
		diffuse = vec3(0.0);
	if (NdotV < 0)
		specular = vec3(0.0);

    return diffuse + specular;
}

// 获取 BRDF 在 L 方向上的概率密度
float pdfBRDF(vec3 V, vec3 N, vec3 L, in HitMaterial material)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);
     
    // 镜面反射 -- 各向同性
    float alpha = max(0.001, sqr(material.roughness));
    float Ds = GTR2(NdotH, alpha);

    // 分别计算三种 BRDF 的概率密度
    float pdf_diffuse = NdotL / M_PI;
    float pdf_specular = Ds * NdotH / (4.0 * dot(L, H));

    // 辐射度统计
    float r_diffuse = (1.0 - material.metallic);
    float r_specular = material.specular;
	if (!material.metallicRoughness)
		r_diffuse = length(material.baseColor);
    float r_sum = r_diffuse + r_specular;

    // 根据辐射度计算选择某种采样方式的概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;
		
	if (NdotL < 0)
		pdf_diffuse = 0.0;
	if (NdotV < 0)
		pdf_specular = 0.0;

    // 根据概率混合 pdf
    float pdf = p_diffuse * pdf_diffuse + p_specular * pdf_specular;

    pdf = max(1e-10, pdf);
    return pdf;
}

// 余弦加权的法向半球采样
vec3 sampleCosineHemisphere(float xi_1, float xi_2, vec3 N)
{
    // 均匀采样 xy 圆盘然后投影到 z 半球
    float r = sqrt(xi_1);
    float theta = xi_2 * 2.0 * M_PI;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - x*x - y*y);

    // 从 z 半球投影到法向半球
    vec3 L = toNormalHemisphere(vec3(x, y, z), N);
    return L;
}

// 按照辐射度分布分别采样三种 BRDF
vec3 sampleBRDF(float xi_1, float xi_2, float xi_3, vec3 V, vec3 N, in HitMaterial material)
{
    float alpha_GTR2 = max(0.001, sqr(material.roughness));
    
    // 辐射度统计
    float r_diffuse = (1.0 - material.metallic);
    float r_specular = material.specular;
	if (!material.metallicRoughness)
		r_diffuse = length(material.baseColor);
    float r_sum = r_diffuse + r_specular;

    // 根据辐射度计算概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;

    // 按照概率采样
    float rd = xi_3;

    // 漫反射
    if (rd <= p_diffuse)
        return sampleCosineHemisphere(xi_1, xi_2, N);
    // 镜面反射
    else
        return sampleGTR2(xi_1, xi_2, V, N, alpha_GTR2);

    return vec3(0, 1, 0);
}

vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}

vec3 applyNormalMap(vec3 geometryNormal, vec4 tangent, vec3 bump)
{
	// Calculate normal in tangent space
	vec3 T = normalize(tangent.xyz);
	if (ORTHOGONALIZE == 1)
		T = normalize(T - dot(geometryNormal, T) * geometryNormal);
	vec3 B = cross(geometryNormal, T) * tangent.w;
	mat3 TBN = mat3(T, B, geometryNormal);
	return TBN * normalize(bump * 2.0 - vec3(1.0));
}