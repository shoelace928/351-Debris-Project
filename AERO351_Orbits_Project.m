%% Company Name:
    % VACUUM - Vehicle And Craft Under Unused Missions
    % Februus - god of purification 
    % Space Custodians
    % Geonitors 
%% AERO351-02 Orbital Debris Clean Up 

%% Station Keeping
clear all; close all; clc; 

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

%% TLE.txt Conversion and Upload
        %first convert TLE into a text file 
%s/c 1 tle
inc1 = tle1(2,3) * (pi/180) ;   %radians, inclination
epoch1 = tle1(1,4) ;    %year and day fraction
RAAN1 = tle1(2,4) * (pi/180) ;  %radians, right ascension of ascending node
ecc1 = tle1(2,5) ;  %eccentricity, divide by factors of 10 to move decimal to front
arg1 = tle1(2,6) * (pi/180) ;   %radians, argument of periapse
Me1 = tle1(2,7) * (pi/180) ;    %radians, mean anomaly at epoch
n1 = tle1(2,8) ;    %mean motion at epoch
orb1 = tle1(2,9) ;  %number of orbit at epoch 

%% Lacey's Functions
    %Lambert's Solver
      function [v1 v2] = lambert(r1,r2, delta_t_given)
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
    
arg = y ;  %argument of periapse in radians
arg_degrees = arg * (180/pi) ;    %argument of periapse in degrees
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
