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


