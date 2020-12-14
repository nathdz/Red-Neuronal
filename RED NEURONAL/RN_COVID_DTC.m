
%PERCEPCIÓN: RED NEURONAL PARA DETECCCIÓN DE COVID
%Natalia Hernández Ramos

clear, close all, clc


%hacemos un vector para el que los datos se consideraron como "normales"
vec_normal=[36	90;
36.1	82;
36.2	84;
36	    67;
36.3	85;
37.1	112;
36.4	90;
36.5	92;
36.7	90;
36.8	60;
];

%hacemos un vector para el que los datos se consideran como "alarmantes"
vec_alarmante=[38	99;
39	92;
36	138;
37	100;
37.3	90;
37.6	120;
38	    95;
38.2	92;
38.4	96;
39	    100;
37.9	120;
36.9	101;
37.5	74;
37.3	79;
37.6	100;
37.4	82;
37.8	137;
37.5	120;
37.2	100;
];

%Hacemos un vector para el que los datos se consideran como "alta
%probabilidad"
vec_altaprob=[38.7	132;
40	135;
40	117;
39.1	128;
39.9	131;
39.8	131;
40	138;
38.1	130;
38	60;
39.2	139;
39.4	129;
37.1	140;
38.2	100;
39.8	132;
38.3	139;
39.1	100;
38.4	98;
37	58;
38.5	136;
35	85;
40	139;
];


%se crean los targets

for j=1:10
T_normal(:,j)=[1,0,0];
end

for j=1:19
T_alarmante(:,j)=[0,1,0];
end

for j=1:21
T_altaprob(:,j)=[0,0,1];
end

%Se genera un vector de entrada y un vector de los targets

input=[vec_normal' vec_alarmante' vec_altaprob'];
targets=[T_normal  T_alarmante  T_altaprob];

red=patternnet(15, 'trainlm')
red. trainParam.epochs=(1000); %Numero de epocas maximas
red. trainParam.max_fail=100; %verifica minimos locales posibles
red. trainParam.min_grad=1e-29; %Error maximo permitido
red. trainParam.mu=0.1; %factor de aprendizaje para modificar los pesos iniciales
red. trainParam.mu_dec=0.1; %Factor de aprendizaje decrecientes
red. trainParam.mu_inc=10; %factor de aprendizaje crecientes
%red.layers{1}.transferFcn='tansig'; %cambiar la funcion de activacion de las neuronas de la capa 1
%red.layers{2}.transferFcn='tansig'; %cambiar la funcion de activacion de las neuronas de la capa 2


%Se divide el set de entrenamiento
configure(red, input, targets);
red.divideParam.trainRatio=50/100
red.divideParam.valRatio=25/100
red.divideParam.testRatio=25/100
[red,tr]=train(red, input, targets); %aquí ocurre el entrenamiento