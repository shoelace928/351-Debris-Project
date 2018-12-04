%% Company Name:
    % VACUUM - Vehicle And Craft Under Unused Missions
    % Februus - god of purification 
    % Space Custodians
    % Geonitors
    
%% AERO351-02 Orbital Debris Clean Up 
%Station Keeping
clear all; close all; clc; 
%Satellite Naming Convention:
%        Lowest to highest in orbit:
%        Sat1 = 
%        Sat2 =
%        Sat3 = 
%        Sat4 = 

%variable names to keep universal in code:
%h - angular momentum
%inc_degrees - inclination in degrees 
    %inc for radians
%ecc - eccentricity
%RAAN_degrees - RAAN in degrees
    %RAAN for radians
%arg_degrees - argument of periapse in degrees
    %arg for radians
%theta_degrees - true anomaly in degrees
    %theta in radians
%Me is mean anomaly
%n is mean motion
%orb is number of orbits
%clarify s/c number with each of the characteristics ex theta1 for s/c 1

%Variables for use throughout:
mu = 398600;
Rearth = 6378; %km

%% Original TLE -> COES -> r and v
% TLE.txt Upload and Conversion
        %tle need to be in txt file
tle = load('Breeze2_tle.txt') ;    %Breeze Rocket Debris at 50.1919 inc in LEO
[inc1, epoch1, RAAN1, ecc1, arg1, Me1, n1] = tle_convert(tle) ;
[irSat1,ivSat1] = TLE_State(RAAN1, arg1, Me1, n1, inc1, ecc1);
iSat1State = [irSat1(1), irSat1(2), irSat1(3), ivSat1(1), ivSat1(2), ivSat1(3)];
tle = load('Breeze1_tle.txt') ;    %Breeze Rocket Debris at 50.0668 inc in LEO
[inc2, epoch2, RAAN2, ecc2, arg2, Me2, n2] = tle_convert(tle) ;
[irSat2,ivSat2] = TLE_State(RAAN2, arg2, Me2, n2, inc2, ecc2);
iSat2State = [irSat2(1), irSat2(2), irSat2(3), ivSat2(1), ivSat2(2), ivSat2(3)];
tle = load('Vanguard1_tle.txt') ;  %Vanguard 1 debris in MEO
[inc3, epoch3, RAAN3, ecc3, arg3, Me3, n3] = tle_convert(tle) ;
[irSat3,ivSat3] = TLE_State(RAAN3, arg3, Me3, n3, inc3, ecc3);
iSat3State = [irSat3(1), irSat3(2), irSat3(3), ivSat3(1), ivSat3(2), ivSat3(3)];
tle = load('Kizuna_tle.txt')  ;    %Kizuna debris in GEO
[inc4, epoch4, RAAN4, ecc4, arg4, Me4, n4] = tle_convert(tle) ;
[irSat4,ivSat4] = TLE_State(RAAN4, arg4, Me4, n4, inc4, ecc4);
iSat4State = [irSat4(1), irSat4(2), irSat4(3), ivSat4(1), ivSat4(2), ivSat4(3)];


%% Original Orbit Propagation:

%FOR EACH USE TLE READER TO PROPAGATE r AND v VECTORS TO TIME AT START:
%Sat1:
tspan1 = [0 1.1205e6];
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat1, Sat1State] = ode45(@twobodymotion, tspan1, iSat1State, options, mu);
rSat1 = Sat1State(:,1:3);
vSat1 = Sat1State(:,4:6);
%Sat2:
tspan2 = [0 1.1206e6];
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat2, Sat2State] = ode45(@twobodymotion, tspan2, iSat2State, options, mu);
rSat2 = Sat2State(:,1:3);
vSat2 = Sat2State(:,4:6);
%Sat3:
tspan3 = [0 .4297e6];
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[ tSat3, Sat3State] = ode45(@twobodymotion, tspan3, iSat3State, options, mu);
rSat3 = Sat3State(:,1:3);
vSat3 = Sat3State(:,4:6);
%Sat4:
tspan4 = [0 1.3819e6];
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat4, Sat4State] = ode45(@twobodymotion, tspan4, iSat4State, options, mu);
rSat4 = Sat4State(:,1:3);
vSat4 = Sat4State(:,4:6);

fin1 = length(rSat1(:,1));
fin2 = length(rSat2(:,1));
fin3 = length(rSat3(:,1));
fin4 = length(rSat4(:,1));

% figure(1)
% hold on
% earth_sphere
% plot3(rSat1(:,1),rSat1(:,2),rSat1(:,3))
% plot3(rSat1(fin1,1), rSat1(fin1,2), rSat1(fin1,3),'o')
% plot3(rSat2(:,1),rSat2(:,2),rSat2(:,3))
% plot3(rSat2(fin2,1), rSat2(fin2,2), rSat2(fin2,3),'o')
% % plot3(rSat3(:,1),rSat3(:,2),rSat3(:,3))
% % plot3(rSat3(fin3,1), rSat3(fin3,2), rSat3(fin3,3),'o')
% plot3(rSat4(:,1),rSat4(:,2),rSat4(:,3))
% plot3(rSat4(fin4,1), rSat4(fin4,2), rSat4(fin4,3),'o')
% legend('Earth', 'Sat1', 'Sat1','Sat2', 'Sat2', 'Sat4', 'Sat4')

%Proof of concept for animating:
% i = 1;
% while i < fin2
%     figure(1)
%     hold on
%     plot3(rSat1(i,1),rSat1(i,2),rSat1(i,3),'.')
%     plot3(rSat2(i,1),rSat2(i,2),rSat2(i,3),'.')
%     plot3(rSat3(i,1),rSat3(i,2),rSat3(i,3),'.')
%     plot3(rSat4(i,1),rSat4(i,2),rSat4(i,3),'.')
%     i = i + 1;
%     pause(.1)
% end

%COES of All Orbits:
orbit1 = rv2coes(rSat1(end,:),vSat1(end,:),mu);
orbit2 = rv2coes(rSat2(end,:),vSat2(end,:),mu);
orbit3 = rv2coes(rSat3(end,:),vSat3(end,:),mu);
orbit4 = rv2coes(rSat4(end,:),vSat4(end,:),mu);

%% ORBIT 1 TO 2

[h1, inc_degrees, N1, RAAN_degrees, ecc1, arg_degrees, theta_degrees] = coes (rSat1(end,:),vSat1(end,:),mu);
a1 = (h1^2/mu)*(1/(1-ecc1^2));
T1 = ((2*pi)/sqrt(mu))*a1^(3/2);
dtOrbit1 = 5*T1;
t = dtOrbit1; %total time counter
%find new r and v vects for each after 5 periods:
[rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, [0 dtOrbit1], mu);

figure(2)
hold on
earth_sphere
plot3(rSat1(:,1),rSat1(:,2),rSat1(:,3))
plot3(rSat1(end,1), rSat1(end,2), rSat1(end,3),'o')
plot3(rSat2(:,1),rSat2(:,2),rSat2(:,3))

%State of 2 after .8*T1:
Sat2State = [rSat2(end,1), rSat2(end,2), rSat2(end,3), vSat2(end,1), vSat2(end,2), vSat2(end,3)];

[tSat2, Sat2State] = ode45(@twobodymotion, t+[0 .9*T1], Sat2State, options, mu);
rSat2 = Sat2State(:,1:3); %location and speed of sat2 at intercept
vSat2 = Sat2State(:,4:6);

[v1, v2] = lambert(rSat1(end,:),rSat2(end,:), .9*T1);

burn1 = v1 - vSat1(end,1:3);
burn2 = v2 - vSat2(end,1:3);

trans1State = [rSat1(end,1), rSat1(end,2), rSat1(end,3), v1(1), v1(2), v1(3)];
[ttrans1, trans1State] = ode45(@twobodymotion, [0 .9*T1], trans1State, options, mu); %find transfer orbit
plot3(rSat2(end,1), rSat2(end,2), rSat2(end,3),'o')

plot3(trans1State(:,1), trans1State(:,2), trans1State(:,3))
legend('Earth', 'Orbit 1', 'SC 1', 'Orbit 2', 'SC 2', 'Transfer Orbit')
title('Transfer Orbit from Satellite 1 to Satellite 2 using Lamberts Solver')
t = t + .9*T1;
dv = norm(burn1)+norm(burn2); %counter for total dv

%% Transfer 2->3
[rSat1, vSat1, ~, ~, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, [0 .9*T1], mu);

[h2, inc_degrees2, N2, RAAN_degrees2, ecc2, arg_degrees2, theta_degrees2] = coes (rSat2(end,:),vSat2(end,:),mu);
a2 = (h2^2/mu)*(1/(1-ecc2^2));
T2 = ((2*pi)/sqrt(mu))*a2^(3/2);
dtOrbit2 = 5*T2;
t = t + dtOrbit2;

[rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, [0 5*T2], mu);

figure(3)
hold on

%hohmann to circ w/ same rp as Sat3:
%find time to periapse:
Sat2state = [rSat2(end,1), rSat2(end,2), rSat2(end,3), vSat2(end,1), vSat2(end,2), vSat2(end,3)];
[tSat2Proj, Sat2ProjState] = ode45(@twobodymotion, [0 T2], Sat2state, options, mu); %find transfer orbit
for ii = 1:length(Sat2ProjState)
    normedSat2State(ii) = norm(Sat2ProjState(ii,:));
end
[rp, timerp] = min(normedSat2State);
ttoperiapse = tSat2Proj(timerp);
t = t +ttoperiapse; 
[rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, [0 ttoperiapse], mu);
%NOW WE ARE AT PERIAPSE OF ORBIT 2
%hohmann time:
ra3 = orbit3.ra;
ecctrans = (ra3-rp)/(ra3+rp);
h_trans = sqrt(rp*mu*(1+ecctrans));
vp_trans = h_trans/rp; %speed at periapse of transfer orbit
va_trans = h_trans/ra3; %speed at apoapse of transfer orbit
a_trans = (orbit2.rp + orbit3.ra)/2;
time_trans = .5*(((2*pi)/sqrt(mu))*a_trans^(3/2));
dv1 = abs(vp_trans-orbit2.vp); %BURN 1
t = t + time_trans; %update t
dv2 = abs(va_trans-sqrt(mu/orbit3.ra)); %BURN 2
dvhohmann = dv1 + dv2; %total delta v of hohmann transfer
dv = dv + dvhohmann;
vscafterburn1 = vSat2(end,:) + dv1*(vSat2(end,:)/norm(vSat2(end,:)));
stateafterburn1 = [rSat2(end,1), rSat2(end,2), rSat2(end,3), vscafterburn1(1), vscafterburn1(2), vscafterburn1(3)];
[timetransEll, transEll] = ode45(@twobodymotion, [0 time_trans], stateafterburn1, options, mu); %find transfer orbit
vtransEll = transEll(:,4:6);
vscafterburn2 = vtransEll(end,:) + dv2*(vtransEll(end,:)/norm(vtransEll(end,:)));
%state vector for our spacecraft immediatly after hohmann burn 2:
circState = [transEll(end,1),transEll(end,2), transEll(end,3), vscafterburn2(1), vscafterburn2(2), vscafterburn2(3)];

[rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, [0 time_trans], mu);
%NOW OUR SC IS CIRCULARIZED AT RA OF ORBIT 3, ALL SATS ARE IN UPDATED
%POSITIONS

 % find common line between planes
N2 = cross(rSat2(1,:), vSat2(1,:))/norm(cross(rSat2(1,:), vSat2(1,:)));
N3 = cross(rSat3(1,:), vSat3(1,:))/norm(cross(rSat3(1,:), vSat3(1,:)));%normal to plane 3
Line23 = cross(N2,N3)/norm(cross(N2,N3));
PlaneChange23 = acosd(dot(N2,N3));

%True Anamolies on the apse line
TA2_apse = acosd(dot(Line23,orbit2.apse));
TA3_apse = acosd(dot(Line23,orbit3.apse));

%correction for goin the long way
crossy = cross(orbit2.apse,Line23);%sweep from perigee(last burn) to node (next burn)
if crossy(3)<0
    TA2_apse = 360 - TA2_apse;
end

%find time from perigee to apse line
time2burn = (TA2_apse*pi/180)/sqrt(mu)*orbit3.ra^(3/2);
t = t + time2burn;
[rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, [0 time2burn], mu);
[ttoplanechng, SCatPlaneChange] = ode45(@twobodymotion, [0 time2burn], circState, options, mu); %find transfer orbit

vSCb4PlaneChange = SCatPlaneChange(end,4:6);
% align the planes (inc and raan change)

C32 = cosd(PlaneChange23)*eye(3) + (1-cos(PlaneChange23))*(Line23')*Line23 - sind(PlaneChange23)*crosss(Line23');
v2 = (C32'*vSCb4PlaneChange')';
dvPlaneChange23 = norm(v2-vSCb4PlaneChange);
dv = dv + dvPlaneChange23;
StateAfterPlaneChange = [SCatPlaneChange(end,1),SCatPlaneChange(end,2),SCatPlaneChange(end,3),v2(1),v2(2),v2(3)];

% FOR VISUAL HELP ONLY
[tPlaneChange, SCAfterPlaneChange] = ode45(@twobodymotion, [0 40000], StateAfterPlaneChange, options, mu); %find transfer orbit
[rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, [0 40000], mu);
figure(6)
hold on
earth_sphere
plot3(SCAfterPlaneChange(:,1),SCAfterPlaneChange(:,2),SCAfterPlaneChange(:,3))
plot3(rSat3(:,1),rSat3(:,2),rSat3(:,3))
plot3(SCatPlaneChange(end,1),SCatPlaneChange(end,2),SCatPlaneChange(end,3),'o')
plot3(rSat3(1,1),rSat3(1,2),rSat3(1,3),'o')

legend('Earth', 'Circ Orbit', 'Orbit3')


% v2 = sqrt(mu/orbit3.ra)*[cosd(PlaneChange23) sind(PlaneChange23)];
% dv = norm(v1-v2);



% 
% earth_sphere
% plot3(rSat2(:,1),rSat2(:,2),rSat2(:,3))
% plot3(rSat2(end,1), rSat2(end,2), rSat2(end,3),'o')
% plot3(rSat3(:,1),rSat3(:,2),rSat3(:,3))
% plot3(rSat3(end,1), rSat3(end,2), rSat3(end,3),'o')
% quiver3(0,0,0, -4538, -7055, 5443)%node line from quinns code
% quiver3(0,0,0, -4682,-6976, 5424)%apse line of 2
% quiver3(0,0,0, -7889, 5685, -2335)%apse line of 3
% legend('Earth', 'Orbit 2', 'SC 2', 'Orbit 3', 'SC 3','node line','apse2','apse3')




%% Lacey's Functions
    %Lambert's Solver
      function [v1, v2] = lambert(r1,r2, delta_t_given)
%given r1 and r2, solve for v1 and v2
mu = 398600 ;
r1_mag = norm(r1)   ;
r2_mag = norm(r2)   ;
crossy = cross(r1,r2) ;
z_comp = crossy(3)   ;
    if z_comp >= 0 
        delta_theta = acos(dot(r1,r2)/(r1_mag*r2_mag)) ; 
    else
        delta_theta = (2*pi) - acos(dot(r1,r2)/(r1_mag*r2_mag)) ;
    end 
A = sin(delta_theta)*sqrt((r1_mag*r2_mag)/(1-cos(delta_theta))) ; 
    if A == 0
        disp('the constant is zero')
    end
    
%z guess bounds and initial conditions
z = 0 ;
zupper = 4*pi^2 ;
zlower = -4*pi^2 ;
ii = 1 ;
delta_t_loop = 10000000 ;
TOL = 10e-8 ;
    %Z VALUE THROUGH BISECTION METHOD
while abs(delta_t_loop - delta_t_given) > TOL 
    %STUMPFF FUNCTIONS
    if z(ii)>0 
        S = ((sqrt(z(ii))) - sin(sqrt(z(ii))))/((sqrt(z(ii)))^3) ;
        C = (1 - cos(sqrt(z(ii))))/z(ii) ;
    elseif z<0
        S = (sinh(sqrt(-z(ii))) - sqrt(-z(ii)))/((sqrt(-z(ii))^3)) ; 
        C = (cosh(sqrt(-z(ii))) - 1)/-z(ii) ;
    else
        S = 1/6 ;
        C = 1/2 ;
    end 
    
    %y, chi, delta_t_loop
    y = r1_mag + r2_mag + ((A*((z*S)-1))/sqrt(C)) ;
    chi = sqrt(y/C) ;  
    delta_t_loop = (((y/C)^(3/2)*S) + (A*sqrt(y)))/sqrt(mu) ;
    
    if delta_t_loop < delta_t_given 
        zlower = z ;
        z = (zupper+zlower)/2 ;
    else
        zupper = z ;
        z = (zupper+zlower)/2 ; 
    end   
end 

%lagrange multipliers
f = 1 - (y/r1_mag) ;
g = A*(sqrt(y/mu)) ;
g_dot = 1 - (y/r2_mag) ;
f_dot = ((f*g_dot) - 1) / g ;

%v1 and v2 
v1 = (1/g)*(r2-(f*r1)) ;
v2 = (f_dot*r1) + (g_dot*v1) ;

      end 
      
    %Non Impulsive Manuever
      function dstatedt = nonimpulsive (t, state, mu, F_t,Isp) 
%function for ode45 proces, defines the differential functions to integrate
m = state(7) ;

dx = state(4) ; %velocity differential equations
dy = state(5) ;
dz = state(6) ;

r = norm([state(1) state(2) state(3)]) ;    %norm of the position vector
v = norm([state(4) state(5) state(6)]) ;    %norm of velocity vector
ddx = ((-mu * state(1)) / r^3)+((F_t*dx)/(m*v)) ;  %Equations of relative motion 
ddy = ((-mu * state(2)) / r^3)+((F_t*dy)/(m*v))  ;
ddz = ((-mu * state(3)) / r^3)+((F_t*dz)/(m*v))  ;

dm = -(F_t*1000)/(Isp*9.8) ;    %kg/s, change in mass as the fuel is burned
dstatedt = [dx;dy;dz;ddx;ddy;ddz;dm] ; 
      end
    
    %Two Body Motion
      function dstatedt = twobodymotion (t, state, mu) 
%function for ode45 proces, defines the differential functions to integrate
dx = state(4) ; %velocity differential equations
dy = state(5) ;
dz = state(6) ;

r = norm([state(1) state(2) state(3)]) ;    %norm of the position vector

ddx = (-mu * state(1)) / r^3 ;  %Equations of relative motion 
ddy = (-mu * state(2)) / r^3 ;
ddz = (-mu * state(3)) / r^3 ;

dstatedt = [dx;dy;dz;ddx;ddy;ddz] ; 
      end

    %Rotation Matrix
      function [Q] = rotmat(arg, inc, RAAN)
%deriving the 3-1-3 rotation matrix for orbits
x = [cos(arg) sin(arg) 0; - sin(arg) cos(arg) 0; 0 0 1] ;
y = [1 0 0; 0 cos(inc) sin(inc); 0 -sin(inc) cos(inc)] ;
z = [cos(RAAN) sin(RAAN) 0; -sin(RAAN) cos(RAAN) 0; 0 0 1] ;

Q = x*y*z ;   %rotation matrix
      end

    %Julian Date
      function [ JD ] = julian( m,d,y,tf )
%julian is a function that will convert UT into Julian Date
%   inputs are month, day, year, and UT time fraction

Jo = 367*y - floor((7*(y+floor((m+9)/12)))/4) + floor((275*m)/9) + d + 1721013.5 ;

JD = Jo + (tf/24) ;

      end
      
    %Classical Orbital Elements
      function [h1, inc_degrees, N1, RAAN_degrees, ecc1, arg_degrees, theta_degrees] = coes (r,v,mu)
%State Vector Magnitude 
r1 = norm(r) ;
v1 = norm(v);

%Angular Momentum
vr = (dot(r,v)) / r1 ;   %radial velocity
h = cross(r,v) ;    %angular momentum
h1 = norm(h) ;

%Inclination
inc = acos(h(3)/h1) ;   %inclination in radians
inc_degrees = inc * (180/pi) ; %inclination in degrees
    while inc_degrees < 0
        inc_degrees =inc_degrees + 180 ;
    end 
    while inc_degrees > 180
        inc_degrees = inc_degrees - 180 ;
    end 
%Node Line
k = [ 0 0 1 ] ;
N = cross(k,h) ;    %node line
N1 = norm(N) ;  %magnitude of node line
x = N(1)/N1 ;

%Right Ascension of the Ascending Node
    if N(2) < 0     
       x = 360 - acosd(x) ;    %quadrant check
    else 
       x = acosd(x) ;
    end
RAAN = x ; %Right Ascension of Ascending Node in radians
RAAN_degrees = RAAN * (180/pi) ;    %RAAN in degrees
    while RAAN_degrees < 0
        RAAN_degrees =RAAN_degrees + 360 ;
    end 
    while RAAN_degrees > 360
        RAAN_degrees = RAAN_degrees - 360 ;
    end 
%eccentricity 
ecc = (cross(v,h) - ((mu*r)/r1)) / mu ; %eccentricity vector
ecc1 = norm(ecc) ;  %eccentricity
y = dot(N,ecc)/(N1*ecc1) ;

%argument of periapse 
    if ecc(3) < 0 
       y = 360 - acosd(y) ;    %quadrant check
       
    else 
       y = acosd(y) ;
    end
    
arg_periapse = y ;  %argument of periapse in radians
arg_degrees = arg_periapse * (180/pi) ;    %argument of periapse in degrees
    while arg_degrees < 0
        arg_degrees = arg_degrees + 360 ;
    end 
    while arg_degrees > 360
        arg_degrees = arg_degrees - 360 ;
    end 
%true anomaly
z = dot(ecc,r)/(ecc1*r1) ;
    if vr < 0 
       z = 360 - acosd(z) ;    %quadrant check
    else 
       z = acosd(z) ;
    end 
theta = z ;    %true anomaly in radians
theta_degrees = theta * (180/pi) ;  %true anomaly in degrees

      end
      
    %TLE Conversion 
      function [inc, epoch, RAAN, ecc, arg, Me, n] = tle_convert(tle)
        inc = tle(2,3) * (pi/180) ;   %radians, inclination
        epoch = tle(1,4) ;    %year and day fraction
        RAAN = tle(2,4) * (pi/180) ;  %radians, right ascension of ascending node
        ecc = tle(2,5)/10e6 ;  %eccentricity, divide by factors of 10 to move decimal to front
        arg = tle(2,6) * (pi/180) ;   %radians, argument of periapse
        Me = tle(2,7) * (pi/180) ;    %radians, mean anomaly at epoch
        n = tle(2,8) ;    %mean motion at epoch 
      end 
      
      function [rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, tspan, mu)
%Propagates orbits over a specific amount of time
%Input all r and v vectors for satellites 1-4 and the tspan and it will
%output all the r and v vectors after that amount of time


iSat1State = [rSat1(end,1), rSat1(end,2), rSat1(end,3), vSat1(end,1), vSat1(end,2), vSat1(end,3)];
iSat2State = [rSat2(end,1), rSat2(end,2), rSat2(end,3), vSat2(end,1), vSat2(end,2), vSat2(end,3)];
iSat3State = [rSat3(end,1), rSat3(end,2), rSat3(end,3), vSat3(end,1), vSat3(end,2), vSat3(end,3)];
iSat4State = [rSat4(end,1), rSat4(end,2), rSat4(end,3), vSat4(end,1), vSat4(end,2), vSat4(end,3)];

options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);

options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat1, Sat1State] = ode45(@twobodymotion, tspan, iSat1State, options, mu);
rSat1 = Sat1State(:,1:3);
vSat1 = Sat1State(:,4:6);
%Sat2:
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat2, Sat2State] = ode45(@twobodymotion, tspan, iSat2State, options, mu);
rSat2 = Sat2State(:,1:3);
vSat2 = Sat2State(:,4:6);
%Sat3:
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[ tSat3, Sat3State] = ode45(@twobodymotion, tspan, iSat3State, options, mu);
rSat3 = Sat3State(:,1:3);
vSat3 = Sat3State(:,4:6);
%Sat4:
tspan4 = [0 1.3819e6];
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat4, Sat4State] = ode45(@twobodymotion, tspan, iSat4State, options, mu);
rSat4 = Sat4State(:,1:3);
vSat4 = Sat4State(:,4:6);
      end

      %% r and v to COEs function
function Orbit = rv2coes(r,v,mu)
%% angular velocity [km^2/s]
h_vec = cross(r,v);
h = norm(h_vec);
Orbit.h = h;

%% Inc
Orbit.inc = acosd(h_vec(3)/h);%inclination
n = cross([0 0 1],h_vec);%normal vector though acending node

%% RAAN
RAAN = acosd(n(1)/norm(n));%Right Angle of Acending Node
if n(1) < 0 %Quadrant Fix
    RAAN = 360 - RAAN;
end
if isnan(RAAN) %deal with /0
    RAAN = 0;
end
Orbit.RAAN = RAAN;

%% Velocity (Radial and Azmuthal)
V_r = dot(r/norm(r),v);
V_az = cross(r/norm(r),v);%???????unconfirmed equation 

%% Ecc
e_vec = (1/mu)*(r.*(norm(v)^2-mu/norm(r)) - (norm(r)*V_r).*v);
ecc = norm(e_vec);
Orbit.ecc = ecc;
Orbit.apse = e_vec/ecc;


%% ArgPeri
ArgP = acosd(dot(n/norm(n),e_vec/ecc));
if e_vec(3) < 0 %Quadrant Fix
    ArgP = 360 - ArgP;
end
if isnan(ArgP) %deal with /0
    ArgP = 0;
end
Orbit.ArgP = ArgP;

%% True Anamoly
TA = acosd(dot(e_vec/ecc,r/norm(r)));
if V_r < 0
    TA = 360 - TA;
end
if isnan(TA)
    TA = 0;
end
Orbit.TA = TA;

%% Specific Energy [km^2/s^2]
Energy = (norm(v)^2/2 - mu/norm(r));
Orbit.Energy = Energy;

%% Semi-Major Axis [km]
a = h^2/mu/(1-ecc^2);%orbit equation special case
Orbit.a = a;

%% Period [seconds]
P = 2*pi*a^(3/2)/sqrt(mu);
Orbit.P = P;

%% Mean Motion [radians/second]
n = 2*pi/P;
Orbit.n = n;

%% t_0 better explaination to come
E_0 = 2*atan(tand(TA/2)/sqrt((1+ecc)/(1-ecc)));%Eccentic Anamoly [radians]
t_0 = (1/n)*(E_0-ecc*sin(E_0));
Orbit.E_0 = E_0;
Orbit.t_0 = t_0;

%% Mean Anomoly [radians]
Me = E_0 + ecc*sin(E_0);
Orbit.Me = Me;

%% Radius and Velocity at Periapse and Apoapse
Orbit.rp = h^2/mu/(1+ecc);
Orbit.ra = h^2/mu/(1-ecc);
Orbit.vp = h/Orbit.rp;
Orbit.va = h/Orbit.ra;

end

function wx = crosss(w)
wx = [0 -w(3) w(2); w(3) 0 -w(1);-w(2) w(1) 0];
end