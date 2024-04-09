#version 450

/*Inputs from vertex shader*/

in vec4 colour;

void main() {
    gl_FragColor = colour;
}