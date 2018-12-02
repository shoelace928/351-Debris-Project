% Coes to RV function
% Quinn Marsh

% inputs
% inclination, RAAN, ecc, ArgPeri, Mean Anamoly, Mean Motion, mu
% Mean Motion in rev's per day

function [r_eci,v_eci,r_pqw,v_pqw] = tle2rv(inc,RAAN,ecc,ArgP,Me,n,mu)
tol = 1e-8;
n = n*2*pi/24/3600;%xonvert to rads/sec
a = (sqrt(mu)/(2*pi*n))^(2/3);
h = sqrt(a*muearth*(1-ecc^2));

%finding True Anamoly from Me
%Newtons iterative scheme to find Eccentic Anamoly
%Guess for E
    if Me<pi
        E = Me+e/2;
    else
        E = Me-e/2;
    end
    
% the newtons iteration
add = 1; %adding
k = 1; %indexing
    while abs(add)>tol && k<1000 %designed to not run more than 1000 interations
        add = (E-ecc*sin(E)-Me)/(1-ecc*cos(E));
        E = E - add;
        k=k+1;
    end
        
TA = 2*atand(sqrt((1+e)/(1-e))*tan(E/2));%true anomaly
   
R = h^2/mu*(1/(1+ecc*cosd(TA)));%calculate radius

%find state in PQW
r_pqw = R*[cosd(TA) sind(TA) 0]';
v_pqw = mu/h*[-sind(TA) ecc+cosd(TA) 0]';

%Rotation Matric from ECI(1) to PQW(2)
% 3-1-3  rot matrix
C1 = [cosd(ArgP) sind(ArgP) 0; -sind(ArgP) cosd(ArgP) 0; 0 0 1];
C2 = [1 0 0; 0 cosd(inc) sind(inc); 0 -sind(inc) cosd(inc)];
C3 = [cosd(RAAN) sind(RAAN) 0; -sind(RAAN) cosd(RAAN) 0; 0 0 1];
Q = C1*C2*C3;

r_eci = Q'*r_pqw;
v_eci = Q'*v_pqw;
end