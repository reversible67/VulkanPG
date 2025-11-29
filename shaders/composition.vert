#version 460

#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outDir;

layout (binding = 10, scalar) uniform UBO
{
	mat4 inverseMVP;
} ubo;

void main()
{
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
	outDir = vec3(ubo.inverseMVP * vec4(vec2(gl_Position.x, -gl_Position.y), 1.0, 1.0));
}
