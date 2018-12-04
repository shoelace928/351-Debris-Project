%% Company Name:
    % VACUUM - Vehicle And Craft Under Unused Missions
    % Februus - god of purification 
    % Space Custodians
    % Geonitors
    
%% AERO351-02 Orbital Debris Clean Up 

%Station Keeping
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
mu = 398600 ; %gravitational constant for earth

%% TLE.txt Upload and Conversion
        %tle need to be in txt file
tle = load('Breeze2_tle.txt') ;    %Breeze Rocket Debris at 50.1919 inc in LEO
[inc1, epoch1, RAAN1, ecc1, arg1, Me1, n1] = tle_convert(tle) ;
[irSat1, ivSat1] = TLE_State(RAAN1,arg1,Me1,n1,inc1,ecc1) ;
iSat1State = [irSat1(1), irSat1(2), irSat1(3), ivSat1(1), ivSat1(2), ivSat1(3)];
tle = load('Breeze1_tle.txt') ;    %Breeze Rocket Debris at 50.0668 inc in LEO
[inc2, epoch2, RAAN2, ecc2, arg2, Me2, n2] = tle_convert(tle) ;
[irSat2, ivSat2] = TLE_State(RAAN2,arg2,Me2,n2,inc2,ecc2) ;
iSat2State = [irSat2(1), irSat2(2), irSat2(3), ivSat2(1), ivSat2(2), ivSat2(3)];
tle = load('Vanguard1_tle.txt') ;  %Iridium debris in MEO
[inc3, epoch3, RAAN3, ecc3, arg3, Me3, n3] = tle_convert(tle) ;
[irSat3, ivSat3] = TLE_State(RAAN3,arg3,Me3,n3,inc3,ecc3) ;
iSat3State = [irSat3(1), irSat3(2), irSat3(3), ivSat3(1), ivSat3(2), ivSat3(3)];
tle = load('Kizuna_tle.txt')  ;    %Kizuna debris in GEO
[inc4, epoch4, RAAN4, ecc4, arg4, Me4, n4] = tle_convert(tle) ;
[irSat4, ivSat4] = TLE_State(RAAN4,arg4,Me4,n4,inc4,ecc4) ;
iSat4State = [irSat4(1), irSat4(2), irSat4(3), ivSat4(1), ivSat4(2), ivSat4(3)];

%launch date: December 6, 2018 0h UT
        m_ld = 12 ;
        d_ld = 6 ;
        y_ld = 2018 ;
        tf_ld = 0 ;
        Jo_ld = 367*y_ld - floor((7*(y_ld+floor((m_ld+9)/12)))/4) + floor((275*m_ld)/9) + d_ld + 1721013.5 ;
        JD_ld = Jo_ld + (tf_ld/24) ; %julian date for launch
        
%tle propagation
        d = [23 23 1 20] ;
        m = [11 11 12 11] ;
        y = [2018 2018 2018 2018] ;
        tf = [.72026911 .75445815 .63652026 .14468353] ;
        Jo = 367.*y - floor((7.*(y+floor((m+9)./12)))./4) + floor((275*m)./9) + d + 1721013.5 ;
        JD = Jo + (tf./24) ; %julian date for launch
        
        delta_t = (JD_ld - JD)*24*3600 ;
        delta_t = [delta_t(3) delta_t(2) delta_t(1) delta_t(4)] ;
        tspan = [1.061e6 1.058e6 3.7589e5 1.3699e6] ;
        
tspan1 = [0 tspan(2)];
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat1, Sat1State] = ode45(@twobodymotion, tspan1, iSat1State, options, mu);
rSat1 = Sat1State(:,1:3);
vSat1 = Sat1State(:,4:6);

%Sat2:
tspan2 = [0 tspan(1)];
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat2, Sat2State] = ode45(@twobodymotion, tspan1, iSat2State, options, mu);
rSat2 = Sat2State(:,1:3);
vSat2 = Sat2State(:,4:6);

%Sat3:
tspan3 = [0 (tspan(3)+(10*3600))];  %where satellite 3 is 10 hours after midnight
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[ tSat3, Sat3State] = ode45(@twobodymotion, tspan1, iSat3State, options, mu);
rSat3 = Sat3State(end,1:3);
vSat3 = Sat3State(end,4:6);
[h3, inc_degrees3, N3, RAAN_degrees3, ecc3, arg_degrees3, theta_degrees3] = coes (rSat3,vSat3,mu) ;
rp3 = ((h3^2)/mu)*(1/(1+ecc3)) ;
ra3 = ((h3^2)/mu)*(1/(1-ecc3)) ;
a3 = (rp3+ra3)/2 ;
T3 = ((2*pi)/sqrt(mu))*(a3^(3/2)) ;

%Sat4:
tspan4 = [ 0 ((5*T3) + tspan(4) + (10*3600))] ;   %where satellite 4 is 10 hours after midnight
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tSat4, Sat4State] = ode45(@twobodymotion, tspan1, iSat4State, options, mu);
rSat4 = Sat4State(end,1:3);
vSat4 = Sat4State(end,4:6);
[h4, inc_degrees4, N4, RAAN_degrees4, ecc4, arg_degrees4, theta_degrees4] = coes (rSat4,vSat4,mu) ;

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
% plot3(rSat3(:,1),rSat3(:,2),rSat3(:,3))
% plot3(rSat3(fin3,1), rSat3(fin3,2), rSat3(fin3,3),'o')
% plot3(rSat4(:,1),rSat4(:,2),rSat4(:,3))
% plot3(rSat4(fin4,1), rSat4(fin4,2), rSat4(fin4,3),'o')

%% Transfer for 3 to 4
%inc/RAAN change
r = norm(rSat4(end,1:3)) ;   %radius of geostationary orbit
v_3 = sqrt(mu/r) ;  %orbital speed of sat 3

alpha = acos((cos(inc3)*cos(inc4)) + (sin(inc3)*sin(inc4)*cos(RAAN4-RAAN3))) ;
delta_v_planet34 = 2*v_3*sin(alpha/2) ;
theta_4 = deg2rad(theta_degrees4) ;
%Phasing 
h_4 = sqrt(2*mu)*sqrt((r^2)/(2*r)) ;    %angular momentum of sat 4
rp_3 = r ;  %chasing orbit, radius of perigee is equal to the radius of the geo orbit
T1 = ((2*pi)/sqrt(mu)) * (r^(3/2)) ;    %period of sat 4
E_4 = 2.*atan((sqrt((1-ecc4)/(1+ecc4)).*tan(theta_4/2))) ;
t_34 = (T1/(2*pi)).*(E_4-(ecc4*sin(E_4))) ;  %flight time for where sat 4 is to get to perigee of sat 3/4 
T2 = T1 - t_34 ;
a_3 = ((T2.*sqrt(mu))./(2*pi)).^(2/3) ;    %semimajor axis of chasing orbit
ra_3 = (2*a_3)-rp_3 ;   %radius of apogee for chasing orbit
h_3 = sqrt(2*mu)*sqrt((ra_3.*rp_3)./(ra_3+rp_3)) ;    %angular momentum of chasing orbit

v4_p = h_4./rp_3 ;   %speed of sat 4 at perigee
v3_p = h_3./rp_3 ;   %speed of sat 3 in perigee in chasing orbit

delta_v_drop = v3_p - v4_p ;
delta_v_return = v4_p - v3_p ;
delta_v_phasing34 = abs(delta_v_drop) + abs(delta_v_return) ;   %total delta v of phasing change
delta_v_34 = delta_v_planet34 + delta_v_phasing34 ; %add delta v for hohmann transfer to this line


function [xx,yy,zz] = earth_sphere(varargin)
%EARTH_SPHERE Generate an earth-sized sphere.
%   [X,Y,Z] = EARTH_SPHERE(N) generates three (N+1)-by-(N+1)
%   matrices so that SURFACE(X,Y,Z) produces a sphere equal to 
%   the radius of the earth in kilometers. The continents will be
%   displayed.
%
%   [X,Y,Z] = EARTH_SPHERE uses N = 50.
%
%   EARTH_SPHERE(N) and just EARTH_SPHERE graph the earth as a 
%   SURFACE and do not return anything.
%
%   EARTH_SPHERE(N,'mile') graphs the earth with miles as the unit rather
%   than kilometers. Other valid inputs are 'ft' 'm' 'nm' 'miles' and 'AU'
%   for feet, meters, nautical miles, miles, and astronomical units
%   respectively.
%
%   EARTH_SPHERE(AX,...) plots into AX instead of GCA.
% 
%  Examples: 
%    earth_sphere('nm') produces an earth-sized sphere in nautical miles
%
%    earth_sphere(10,'AU') produces 10 point mesh of the Earth in
%    astronomical units
%
%    h1 = gca;
%    earth_sphere(h1,'mile')
%    hold on
%    plot3(x,y,z)
%      produces the Earth in miles on axis h1 and plots a trajectory from
%      variables x, y, and z
%   Clay M. Thompson 4-24-1991, CBM 8-21-92.
%   Will Campbell, 3-30-2010
%   Copyright 1984-2010 The MathWorks, Inc. 
%% Input Handling
[cax,args,nargs] = axescheck(varargin{:}); % Parse possible Axes input
error(nargchk(0,2,nargs)); % Ensure there are a valid number of inputs
% Handle remaining inputs.
% Should have 0 or 1 string input, 0 or 1 numeric input
j = 0;
k = 0;
n = 50; % default value
units = 'km'; % default value
for i = 1:nargs
    if ischar(args{i})
        units = args{i};
        j = j+1;
    elseif isnumeric(args{i})
        n = args{i};
        k = k+1;
    end
end
if j > 1 || k > 1
    error('Invalid input types')
end
%% Calculations
% Scale factors
Scale = {'km' 'm'  'mile'            'miles'           'nm'              'au'                 'ft';
         1    1000 0.621371192237334 0.621371192237334 0.539956803455724 6.6845871226706e-009 3280.839895};
% Identify which scale to use
try
    myscale = 6378.1363*Scale{2,strcmpi(Scale(1,:),units)};
catch %#ok<*CTCH>
    error('Invalid units requested. Please use m, km, ft, mile, miles, nm, or AU')
end
     
% -pi <= theta <= pi is a row vector.
% -pi/2 <= phi <= pi/2 is a column vector.
theta = (-n:2:n)/n*pi;
phi = (-n:2:n)'/n*pi/2;
cosphi = cos(phi); cosphi(1) = 0; cosphi(n+1) = 0;
sintheta = sin(theta); sintheta(1) = 0; sintheta(n+1) = 0;
x = myscale*cosphi*cos(theta);
y = myscale*cosphi*sintheta;
z = myscale*sin(phi)*ones(1,n+1);
%% Plotting
if nargout == 0
    cax = newplot(cax);
    % Load and define topographic data
    load('topo.mat','topo','topomap1');
    % Rotate data to be consistent with the Earth-Centered-Earth-Fixed
    % coordinate conventions. X axis goes through the prime meridian.
    % http://en.wikipedia.org/wiki/Geodetic_system#Earth_Centred_Earth_Fixed_.28ECEF_or_ECF.29_coordinates
    %
    % Note that if you plot orbit trajectories in the Earth-Centered-
    % Inertial, the orientation of the contintents will be misleading.
    topo2 = [topo(:,181:360) topo(:,1:180)]; %#ok<NODEF>
    
    % Define surface settings
    props.FaceColor= 'texture';
    props.EdgeColor = 'none';
    props.FaceLighting = 'phong';
    props.Cdata = topo2;
    % Create the sphere with Earth topography and adjust colormap
    surface(x,y,z,props,'parent',cax)
    colormap(topomap1)
% Replace the calls to surface and colormap with these lines if you do 
% not want the Earth's topography displayed.
%     surf(x,y,z,'parent',cax)
%     shading flat
%     colormap gray
    
    % Refine figure
    axis equal
    xlabel(['X [' units ']'])
    ylabel(['Y [' units ']'])
    zlabel(['Z [' units ']'])
    view(127.5,30)
else
    xx = x; yy = y; zz = z;
end
end
    %tle to state
    function [R,V] = TLE_State(RAAN,OMEGA,ME,MM,INC,ecc)

%Given: RAAN,OMEGA = AoP, ME (Mean Anomaly), Mean Motion = MM, Inc, ecc 
muearth = 398600;
tol = 10^-8;
MM = MM *1/(60^2*24)*2*pi;
a = (muearth/(MM^2))^(1/3);
h = sqrt(a*muearth*(1-ecc^2));
 

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
   %TA = rad2deg(TA1);
   
R_peri = (h^2/muearth)*(1/(1+ecc*cos(TA)))*[cos(TA) sin(TA) 0]; % in the p hat direction 
V_peri = (muearth/h) * [-sin(TA) ecc+cos(TA) 0]; % in the q hat direction

QxX = [-sin(RAAN)*cos(INC)*sin(OMEGA)+cos(RAAN)*cos(OMEGA)...
-sin(RAAN)*cos(INC)*cos(OMEGA)-cos(RAAN)*sin(OMEGA) sin(RAAN)...
*sin(INC); cos(RAAN)*cos(INC)*sin(OMEGA)+sin(RAAN)*cos(OMEGA)...
cos(RAAN)*cos(INC)*cos(OMEGA)-sin(RAAN)*sin(OMEGA) -cos(RAAN)...
*sin(INC); sin(INC)*sin(OMEGA) sin(INC)*cos(OMEGA) cos(INC)];

R = transpose(QxX*transpose(R_peri));
V = transpose(QxX*transpose(V_peri));


    end 
    %Propagation
    function [rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, tspan, mu)
%Propagates orbits over a specific amount of time
%Input all r and v vectors for satellites 1-4 and the tspan and it will
%output all the r and v vectors after that amount of time


iSat1State = [rSat1(1), rSat1(2), rSat1(3), vSat1(1), vSat1(2), vSat1(3)];
iSat2State = [rSat2(1), rSat2(2), rSat2(3), vSat2(1), vSat2(2), vSat2(3)];
iSat3State = [rSat3(1), rSat3(2), rSat3(3), vSat3(1), vSat3(2), vSat3(3)];
iSat4State = [rSat4(1), rSat4(2), rSat4(3), vSat4(1), vSat4(2), vSat4(3)];

options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
tspan1 = [ 0 tspan(3) ] ;
[tSat1, Sat1State] = ode45(@twobodymotion, tspan1, iSat1State, options, mu);
rSat1 = Sat1State(:,1:3);
vSat1 = Sat1State(:,4:6);

tspan2 = [ 0 tspan(2) ] ;
[tSat2, Sat2State] = ode45(@twobodymotion, tspan2, iSat2State, options, mu);
rSat2 = Sat2State(:,1:3);
vSat2 = Sat2State(:,4:6);

tspan3 = [ 0 tspan(1) ] ;
[tSat3, Sat3State] = ode45(@twobodymotion, tspan3, iSat3State, options, mu);
rSat3 = Sat3State(:,1:3);
vSat3 = Sat3State(:,4:6);

tspan4 = [ 0 tspan(4) ] ;
[tSat4, Sat4State] = ode45(@twobodymotion, tspan4, iSat4State, options, mu);
rSat4 = Sat4State(:,1:3);
vSat4 = Sat4State(:,4:6);

% rSat1 = rSat1(end,1:3);
% vSat1 = vSat1(end,1:3);
% rSat2 = rSat2(end,1:3);
% vSat2 = vSat2(end,1:3);
% rSat3 = rSat3(end,1:3);
% vSat3 = vSat3(end,1:3);
% rSat4 = rSat4(end,1:3);
% vSat4 = vSat4(end,1:3);

end

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
    while theta_degrees < 0
        theta_degrees = theta_degrees + 360 ;
    end 
    while theta_degrees > 360
        theta_degrees = theta_degrees - 360 ;
    end 

      end
      
    %TLE Conversion 
      function [inc, epoch, RAAN, ecc, arg, Me, n] = tle_convert(tle)
        inc = tle(2,3) * (pi/180) ;   %radians, inclination
        epoch = num2str(tle(1,4)) ;    %year and day fraction
        RAAN = tle(2,4) * (pi/180) ;  %radians, right ascension of ascending node
        ecc = tle(2,5)/10e6 ;  %eccentricity, divide by factors of 10 to move decimal to front
        arg = tle(2,6) * (pi/180) ;   %radians, argument of periapse
        Me = tle(2,7) * (pi/180) ;    %radians, mean anomaly at epoch
        n = tle(2,8) ;    %mean motion at epoch 
      end 
      
       
        
       

















        
        
        