#version 450 core

#define SHC_0 0.28209479177387814f

in vec3 vCenter;
in vec3 vColour;
in float vOpacity;
out vec4 colour;

uniform mat4 mvp;

vec3 colour_from_harmonics() {
    return (0.5 + SHC_0 * vColour);
}

void main() {
	gl_PointSize = 5;
	gl_Position = mvp * mat4(-1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1)* vec4(vCenter, 1);
	vec3 fColour = colour_from_harmonics();
	colour = vec4(fColour, vOpacity);
}
