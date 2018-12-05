clc; clear all; close all

muearth = 398600;
tle1 = load('Breeze1_tle.txt');
tle2 = load('Breeze2_tle.txt');
tle3 = load('Vanguard1_tle.txt');
tle4 = load('Kizuna_tle.txt');

 [inc1, epoch1, RAAN1, ecc1, arg1, Me1, n1] = tle_convert(tle1);
 [inc2, epoch2, RAAN2, ecc2, arg2, Me2, n2] = tle_convert(tle2);
 [inc3, epoch3, RAAN3, ecc3, arg3, Me3, n3] = tle_convert(tle3);
 [inc4, epoch4, RAAN4, ecc4, arg4, Me4, n4] = tle_convert(tle4);

 
 [R1,V1] = TLE_State(RAAN1,arg1,Me1,n1,inc1,ecc1); 
 [R2,V2] = TLE_State(RAAN2,arg2,Me2,n2,inc2,ecc2); 
 [R3,V3] = TLE_State(RAAN3,arg3,Me3,n3,inc3,ecc3); 
 [R4,V4] = TLE_State(RAAN4,arg4,Me4,n4,inc4,ecc4); 
 
Breeze_1 = 18327.72026911000;

Breeze_2 = 18327.75445815;
Vanguard_1 = 18335.64941051;
Kizuna = 18324.14468353;


Start = 18340;

dDay1 = Start - Breeze_1;
dDay2 = Start - Breeze_2;
dDay3 = Start - Vanguard_1;
dDay4 = Start - Kizuna;

dSec1 = dDay1 * 24*60*60;
dSec2 = dDay2 * 24*60*60;
dSec3 = dDay3 * 24*60*60;
dSec4 = dDay4 * 24*60*60;


 
 tspan1 = [0 dSec1];
 tspan2 = [0 dSec2];
 tspan3 = [0 dSec3];
 tspan4 = [0 dSec4];
 
[rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(R1, V1, R2, V2, R3, V3, R4, V4, tspan1,tspan2,tspan3,tspan4, 398600);

figure(1)
hold on
plot3(rSat1(:,1),rSat1(:,2),rSat1(:,3))
plot3(rSat1(end,1),rSat1(end,2),rSat1(end,3),'ro')
plot3(rSat2(:,1),rSat2(:,2),rSat2(:,3))
plot3(rSat2(end,1),rSat2(end,2),rSat2(end,3),'ro')
plot3(rSat3(:,1),rSat3(:,2),rSat3(:,3))
plot3(rSat3(end,1),rSat3(end,2),rSat3(end,3),'ro')
plot3(rSat4(:,1),rSat4(:,2),rSat4(:,3))
plot3(rSat4(end,1),rSat4(end,2),rSat4(end,3),'ro')

earth_sphere

% r1 = rSat1(end,1:3);
% v1 = vSat1(end,1:3);
% r2 = rSat2(end,1:3);
% v2 = vSat2(end,1:3);
% r3 = rSat3(end,1:3);
% v3 = vSat3(end,1:3)
% r4 = rSat4(end,1:3);
% v4 = vSat4(end,1:3)
% 
% [ a1,H1, Ecc1, Inc1, raan1, Arg1, Theta1 ] = COES_JP( v1,r1,muearth );
% [ a2,H2, Ecc2, Inc2, raan2, Arg2, Theta2 ] = COES_JP( v2,r2,muearth );
% [ a3,H3, Ecc3, Inc3, raan3, Arg3, Theta3 ] = COES_JP( v3,r3,muearth );
% [ a4,H4, Ecc4, Inc4, raan4, Arg4, Theta4 ] = COES_JP( v4,r4,muearth );

% COES(1,:) = [a1,a2,a3,a4];
% COES(2,:) = [Ecc1,Ecc2,Ecc3,Ecc4];
% COES(3,:) = [Inc1 Inc2 Inc3 Inc4];
% COES(4,:) = [raan1,raan2,raan3,raan4];
% COES(5,:) = [Arg1,Arg2,Arg3,Arg4];
% COES(6,:) = [Theta1 Theta2 Theta3 Theta4];

%dlmwrite('COES_Start.txt',COES)

%% Transfer 3 to Sat4

R_Sat3i = 1.0e+03*[-2.8626, -6.7371, 4.9364];
V_Sat3i = [5.7034, -3.2120, 1.0978];
R_Sat4i = 1.0e+04*[-3.3365, -2.5746, 0.1092];
V_Sat4i = [1.8771, -2.4349, -0.0680];

%States at start time (Dec 6, 2018 @ 0:00:00)
State3i = [R_Sat3i, V_Sat3i]; 
State4i = [R_Sat4i, V_Sat4i]; 

%Start time to 40 hours later
tspan34i = [0 40*60*60]; 

%ODE to propagate to 40 hours after start time
opt = odeset('RelTol',1e-8,'AbsTol',1e-8);
figure(2)
[t3_40,State3_40] = ode45(@Aero351twobodymotion,tspan34i,State3i,opt,muearth);
[t4_40,State4_40] = ode45(@Aero351twobodymotion,tspan34i,State4i,opt,muearth);

figure(2)
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
hold on
plot3(State3_40(:,1),State3_40(:,2),State3_40(:,3))
plot3(State3_40(end,1),State3_40(end,2),State3_40(end,3),'ro')
plot3(State4_40(:,1),State4_40(:,2),State4_40(:,3))
plot3(State4_40(end,1),State4_40(end,2),State4_40(end,3),'bo')
earth_sphere
title('Transfer 3 of SC to Rendezvous with Kizuna Satellite ')
%legend('Orbit 3','Space Craft')%,'Orbit4','Kizuna Satellite'
%R and V of SC and debris 4 after 40 hours from start time
R3_40 = State3_40(end,(1:3));
R4_40 = State4_40(end,(1:3));
V3_40 = State3_40(end,(4:6));
V4_40 = State4_40(end,(4:6));

%COES
[ a3,H3, Ecc3, Inc3, raan3, Arg3, Theta3_40 ] = COES_JP( V3_40,R3_40,muearth );
[ a4,H4, Ecc4, Inc4, raan4, Arg4, Theta4_40 ] = COES_JP( V4_40,R4_40,muearth );

% Transfer Calcs
Rp3 = H3^2/muearth * (1/(1+Ecc3));
Rp_t = Rp3;
Ra_t = a4;

ecc_t = (Ra_t - Rp_t)/(Ra_t + Rp_t);
H_t = sqrt(Rp_t*muearth*(1+ecc_t));
a_t = (Ra_t + Rp_t)/2;
T_t = 2*pi/sqrt(muearth)*(a_t^(3/2));

% N3 and N4 = rev/sec so T3 and T4 are seconds/rev

T3 = (2*pi/sqrt(muearth))*a3^(3/2);
T4 = (2*pi/sqrt(muearth))*a4^(3/2);
N3 = 2*pi/T3;
N4 = 2*pi/T4;

%Calculate Time Since Periapse of SC in orbit 3
[ Time_since_Rp3,~ ] = TS_Periapse( 0,deg2rad(Theta3_40),ecc3, N3);

%Calculate time to periapse 
Time_to_Rp3 = T3 - (Time_since_Rp3*60);
Tspan_toRp3 = [0 Time_to_Rp3];
%***** I know we could just subtract Time since periapse from 40 hours but
%id rather be safe and increase the time before transfer**********
opt = odeset('RelTol',1e-8,'AbsTol',1e-8);
opt1 = odeset('RelTol',1e-8,'AbsTol',1e-8);%,'MaxStep',20
[t3_peri,State3_peri] = ode45(@Aero351twobodymotion,Tspan_toRp3,State3_40(end,(1:6)),opt,muearth);
[t4_peri3,State4_peri3] = ode45(@Aero351twobodymotion,Tspan_toRp3,State4_40(end,(1:6)),opt,muearth);

R3_Rp3 = State3_peri(end,(1:3));
R4_Rp3 = State4_peri3(end,(1:3));
V3_Rp3 = State3_peri(end,(4:6));
V4_Rp3 = State4_peri3(end,(4:6));

%Check Theta of Periapse
[ ~,~, ~, ~, ~, ~, Theta3_peri ] = COES_JP( V3_Rp3,R3_Rp3,muearth );

curve = animatedline('LineWidth',1.5);
curve2 = animatedline('LineWidth',1.5);

for i =1:length(State4_peri3)
%     if i == 1
% addpoints(curve,State3_peri(i,1),State3_peri(i,2),State3_peri(i,3))
% head = plot3(State3_peri(i,1),State3_peri(i,2),State3_peri(i,3),'ro');
% drawnow
% delete(head)
%     else
addpoints(curve,State3_peri(i*5,1),State3_peri(i*5,2),State3_peri(i*5,3))
head = plot3(State3_peri(i*5,1),State3_peri(i*5,2),State3_peri(i*5,3),'ro');
drawnow
delete(head)
%     end
addpoints(curve2,State4_peri3(i,1),State4_peri3(i,2),State4_peri3(i,3))
head_4 = plot3(State4_peri3(i,1),State4_peri3(i,2),State4_peri3(i,3),'bo');
drawnow
delete(head_4)
view([155 11])
end

%Calculate the V of transfer at Rp3
V_peri3_t = muearth/H_t * [-sind(Theta3_peri) ecc_t+cosd(Theta3_peri) 0];

[ QxX3,~ ] = QTransform( raan3,Inc3,Arg3 );

V_geo3_t = transpose(QxX3*transpose(V_peri3_t));

% DV from orbit 3 to start of Hohmann
dV3_to_trans = norm(V3_Rp3 - V_geo3_t);

% Propagate to end of Hohmann
tspan_trans = [0 T_t/2];
Statei_trans = [R3_Rp3, V_geo3_t];
opt2 = odeset('RelTol',1e-8,'AbsTol',1e-8,'MaxStep',490);
[t3_trans,State_trans] = ode45(@Aero351twobodymotion,tspan_trans,Statei_trans,opt2,muearth);
[t4_trans,State4_trans] = ode45(@Aero351twobodymotion,tspan_trans,State4_peri3(end,(1:6)),opt2,muearth);

for i =1:length(State4_trans)
    if i ==1
addpoints(curve,State_trans(i,1),State_trans(i,2),State_trans(i,3))
head2 = plot3(State_trans(i,1),State_trans(i,2),State_trans(i,3),'ro');
drawnow
delete(head2)
    else
addpoints(curve,State_trans((i-1)*2,1),State_trans((i-1)*2,2),State_trans((i-1)*2,3))
head2 = plot3(State_trans((i-1)*2,1),State_trans((i-1)*2,2),State_trans((i-1)*2,3),'ro');
drawnow
delete(head2)
    end
addpoints(curve2,State4_trans(i,1),State4_trans(i,2),State4_trans(i,3))
head24 = plot3(State4_trans(i,1),State4_trans(i,2),State4_trans(i,3),'bo');
drawnow
delete(head24)
view([155 11])
end

% Check Position after Hohmann Transfer
Rt_end = norm(State_trans(end,(1:3)));

R3_tf = State_trans(end,(1:3));
V3_tf = State_trans(end,(4:6));

[ ~,~, ~, ~, ~, ~, Theta3f_trans ] = COES_JP( V3_tf,R3_tf,muearth );
%Circle Calcs
a3_circ = Rt_end;
V3_circ_mag = sqrt(muearth/a3_circ);

%Calc V_vect for circle
H3_circ = sqrt(Rt_end*muearth);
V_peri3_circ = muearth/H3_circ * [-sind(Theta3f_trans) ecc4+cosd(Theta3f_trans) 0];

V3_circ = transpose(QxX3*transpose(V_peri3_circ));

dV3_to_Circle = norm(V3_tf - V3_circ);

% Go through Circle till intersection point
tspan3_circ = [0 1.851*60*60];
State3i_circ = [R3_tf,V3_circ];
opt3 = odeset('RelTol',1e-8,'AbsTol',1e-8);%,'MaxStep',30,'Refine',10
[t3_circ,State3_circ] = ode45(@Aero351twobodymotion,tspan3_circ,State3i_circ,opt3,muearth);
[t4_circ,State4_circ] = ode45(@Aero351twobodymotion,tspan3_circ,State4_trans(end,(1:6)),opt3,muearth);
% 
for i =1:length(State3_circ)
addpoints(curve,State3_circ(i,1),State3_circ(i,2),State3_circ(i,3))
head3 = plot3(State3_circ(i,1),State3_circ(i,2),State3_circ(i,3),'ro');
drawnow
delete(head3)

addpoints(curve2,State4_circ(i,1),State4_circ(i,2),State4_circ(i,3))
head34 = plot3(State4_circ(i,1),State4_circ(i,2),State4_circ(i,3),'bo');
drawnow
delete(head34)
end


%Calculate inc and Raan Change
a4 ;   %radius of geostationary orbit

alpha = acos((cos(inc3)*cos(inc4)) + (sin(inc3)*sin(inc4)*cos(RAAN4-RAAN3))) ;
dV_plane34 = 2*norm(V3_circ)*sin(alpha/2);

%Find intersection points between 3 and 4
% 
% for i =1:136
% addpoints(curve2,State4_40(i,1),State4_40(i,2),State4_40(i,3))
% head4 = plot3(State4_40(i,1),State4_40(i,2),State4_40(i,3),'ro');
% drawnow
% delete(head4)
% view([155 11])
% end

% Velocity of SC at intersection point of orbit 4
V3f_Circ = State3_circ(end,(4:6));

% Velocity of SC to change to orbit 4
V3_Plane4 = State4_40(136,(4:6));

%Check if value for dV is correct
dV_plane34_check = norm(V3f_Circ -V3_Plane4);

% Put SC in orbit 4
R3i_Plane4 = State3_circ(end,(1:3));
State3i_Orbit4 = [R3i_Plane4,V3_Plane4];
plot3(R3i_Plane4(1),R3i_Plane4(2),R3i_Plane4(3),'rx');

% for i =1:length(State3_orbit4)
% addpoints(curve,State3_orbit4(i,1),State3_orbit4(i,2),State3_orbit4(i,3))
% head4 = plot3(State3_orbit4(i,1),State3_orbit4(i,2),State3_orbit4(i,3),'ro');
% drawnow
% delete(head4)
% view([155 11])
% end

T_from40 = Time_to_Rp3 + T_t/2 + (1.852*60*60);
tspan_from40 = [0 T_from40];
State4i_from40 = [R4_40,V4_40];
[t4_from40,State4_from40] = ode45(@Aero351twobodymotion,tspan_from40,State4i_from40,opt,muearth);

R4_from40 = State4_from40(end,(1:3));
V4_from40 = State4_from40(end,(4:6));

% for i =1:length(State4_from40)
% addpoints(curve2,State4_from40(i,1),State4_from40(i,2),State4_from40(i,3))
% head4 = plot3(State4_from40(i,1),State4_from40(i,2),State4_from40(i,3),'ro');
% drawnow
% delete(head4)
% view([155 11])
% end

[ ~,~, ~, ~, ~, ~, Theta4_from40 ] = COES_JP(V4_from40,R4_from40,muearth);
[ ~,~, ~, ~, ~, ~, Theta3_from40 ] = COES_JP(V3_Plane4,State4_40(136,(1:3)),muearth);

% Find Time since Periapse of SC and Object 4
[ TR4_since_Rp4,~ ] = TS_Periapse( 0,deg2rad(Theta4_from40),ecc4, N4);
[ TR3_since_Rp4,~ ] = TS_Periapse( 0,deg2rad(Theta3_from40),ecc4, N4);

dT_34 = abs(TR3_since_Rp4 - TR4_since_Rp4); %in minutes

T_phasing = T4 + (dT_34*60);
Rp_phase = norm(R3i_Plane4);
a_phase = (T_phasing*sqrt(muearth)/(2*pi))^(2/3);
Ra_phase = 2*a_phase - Rp_phase;
ecc_phase = (Ra_phase-Rp_phase)/(Ra_phase+Rp_phase);
h_phase = sqrt(Rp_phase*(muearth*(1+ecc_phase)));

V3_peri_phase = muearth/h_phase * [-sind(0) ecc_phase+cosd(0) 0];

[ QxX4,~ ] = QTransform( raan3,Inc4,Arg4 );
V3_phase = transpose(QxX4*transpose(V3_peri_phase));

dV3_PrePhase = norm(V3_Plane4 - V3_phase);
State3i_phase = [R3i_Plane4,V3_phase];

tspan_phase = [0 T_phasing];

opt3 = odeset('RelTol',1e-8,'AbsTol',1e-8,'MaxStep',300);%
[t3_phase,State3_phase] = ode45(@Aero351twobodymotion,tspan_phase,State3i_phase,opt3,muearth);
[t4_phase,State4_phase] = ode45(@Aero351twobodymotion,tspan_phase,State4_from40(end,:),opt3,muearth);


for i =1:2:length(State3_phase)
addpoints(curve,State3_phase(i,1),State3_phase(i,2),State3_phase(i,3))
head4 = plot3(State3_phase(i,1),State3_phase(i,2),State3_phase(i,3),'ro');
drawnow
delete(head4)

addpoints(curve2,State4_phase(i,1),State4_phase(i,2),State4_phase(i,3))
head44 = plot3(State4_phase(i,1),State4_phase(i,2),State4_phase(i,3),'bo');
drawnow 
delete(head44)
% view([148 7])
end


V3_PostPhase = State3_phase(end,(4:6));
V4_PostPhase = State4_phase(end,(4:6));

dV_PostPhase = norm(V3_PostPhase - V4_PostPhase);

dV34_total = dV3_to_trans + dV3_to_Circle + dV_plane34 + dV3_PrePhase + dV_PostPhase;
T_total_s = 40*60*60 + Time_to_Rp3 + T_t/2 + tspan3_circ(2) + T_phasing + T4*5 %seconds
T_total_hours = T_total_s/3600 %hours
%% functions
function [ a,H, Ec, Inc, RAAN, AoP, TA ] = COES_JP( V,R,mu )
%Semi-Major Axis
MV = norm(V); %Turns vector into magnitude
MR = norm(R);
E = (MV^2) / 2 - mu/MR;
a = -mu/(2*E); %calculates the semi major axis
%eccentricity
e = (1/mu) * (((MV^2 - (mu/MR))* R) - (dot(R,V)*V)); %Vector for Eccentricity shown in workspace
Ec = norm(e); %Turns e vector into magnitude
%Inclination (NO QUADRANT CHECK)
h = cross(R,V); %Takes cross product of R and V
H = norm(h);
Inc = acosd(h(3)/H); %Takes the inverse cosine in degrees
%RAAN
k = [0,0,1]; % k unit vector
n = cross(k,h);
N=norm(n);
RAAN = acosd(n(1)/N);
if n(2) < 0 %Quadrant Check for RAAN
 RAAN = 360 - RAAN;
elseif n(2) > 0
 RAAN = RAAN;
else n(2) = 0 ;
 RAAN = 'none' ;
end
%Argument of Perigee
AoP = acosd(dot(n,e)/(N * Ec)) ;
check_aop = dot(n,e);
if e(3) < 0 %Quadrant Check for AoP
 AoP = 360 - AoP;
elseif e(3) > 0
 Aop = AoP;
else Ec = 0 ;
 AoP = 'none' ;
end
%True Anomaly
TA = acosd(dot(e,R)/(Ec * MR));
if dot(R,V) > 0 %Quadrant Check for True Anomaly
 TA = TA;
elseif Ec == 0
 TA = 'none' ;
else
 TA = 360 - TA;
end
end

%%
function [ Time,TA ] = TS_Periapse( time,ta,ecc,N )
% **If you want to calculate time set time=0
% **If you want to calculate TA set ta=0 (Output is in radians)
% **Have not included other orbit types**

    
    tol = 1e-8; % sets tolerance
    if (ecc<1) && (ecc>0) % conditions for calculating correct
                          % item (Time or True Anomaly) (using equations from lecture)
        
    if time == 0 % conditions for calculating time if TA given
        E = 2 * atan((sqrt((1-ecc)/(1+ecc))*tan(ta/2)));
        ME = E - ecc*sin(E);
        Time = (ME/N)/60; % Output in minutes
        TA = NaN;
    end
    
    if ta == 0 % condition for calculating True Anomaly if time is given
        ME = N*time;
        
        if ME < pi % finding correct guess for E
            E = ME + ecc/2;
        elseif ME > pi 
            E = ME - ecc/2;
        end
        
        ratio = 1; % sets initial ratio
        n = 1; % sets initial iteration value
        
        while ratio(end) > tol % since ratio values are being stored, end us used
            ratio(n) = (E(n) - ecc*sin(E(n)) - ME)/(1 - ecc*cos(E(n))); % ratio calc
            E(n+1) = E(n) - ratio(n); % Calculating E 
            n = n+1; % increasing iteration value
        end
        
   tantheta2 = (sqrt((1+ecc)/(1-ecc))) * tan((E(end)/2)); % goes into true anomaly calc
   TA = atan(tantheta2) * 2; % calculates true anomaly (radians)
   Time = NaN; % Time NaN because time is given to find TA for this part
    end
    
    end
    
end
%%
function [ QxX,QXx ] = QTransform( RAAN,INC,OMEGA )

% Creates Rotation Matrix for the transformation between Perifocal and
% Geocentric Components

QxX = [-sind(RAAN)*cosd(INC)*sind(OMEGA)+cosd(RAAN)*cosd(OMEGA)...
-sind(RAAN)*cosd(INC)*cosd(OMEGA)-cosd(RAAN)*sind(OMEGA) sind(RAAN)...
*sind(INC); cosd(RAAN)*cosd(INC)*sind(OMEGA)+sind(RAAN)*cosd(OMEGA)...
cosd(RAAN)*cosd(INC)*cosd(OMEGA)-sind(RAAN)*sind(OMEGA) -cosd(RAAN)...
*sind(INC); sind(INC)*sind(OMEGA) sind(INC)*cosd(OMEGA) cosd(INC)];

QXx = transpose(QxX);
end

%%
function [ dstatedt ] = Aero351twobodymotion( t,state,mue )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
dx = state(4);
dy = state(5);
dz = state(6);

r = norm([state(1) state(2) state(3)]);

ddx = -mue*state(1)/r^3;
ddy = -mue*state(2)/r^3;
ddz = -mue*state(3)/r^3;

dstatedt = [dx;dy;dz;ddx;ddy;ddz];
end















