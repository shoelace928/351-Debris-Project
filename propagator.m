function [rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4] = propagator(rSat1, vSat1, rSat2, vSat2, rSat3, vSat3, rSat4, vSat4, tspan, mu)
%Propagates orbits over a specific amount of time
%Input all r and v vectors for satellites 1-4 and the tspan and it will
%output all the r and v vectors after that amount of time


iSat1State = [rSat1(1), rSat1(2), rSat1(3), vSat1(1), vSat1(2), vSat1(3)];
iSat2State = [rSat2(1), rSat2(2), rSat2(3), vSat2(1), vSat2(2), vSat2(3)];
iSat3State = [rSat3(1), rSat3(2), rSat3(3), vSat3(1), vSat3(2), vSat3(3)];
iSat4State = [rSat4(1), rSat4(2), rSat4(3), vSat4(1), vSat4(2), vSat4(3)];

options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);

[Sat1State, tSat1] = ode45(@twobodymotion, tspan, iSat1State, options, mu);
rSat1 = Sat1State(:,1:3);
vSat1 = Sat1State(:,4:6);

[Sat2State, tSat2] = ode45(@twobodymotion, tspan, iSat2State, options, mu);
rSat2 = Sat2State(:,1:3);
vSat2 = Sat2State(:,4:6);

[Sat3State, tSat3] = ode45(@twobodymotion, tspan, iSat3State, options, mu);
rSat3 = Sat3State(:,1:3);
vSat3 = Sat3State(:,4:6);

[Sat4State, tSat4] = ode45(@twobodymotion, tpan, iSat4State, options, mu);
rSat4 = Sat4State(:,1:3);
vSat4 = Sat4State(:,4:6);

fin = length(rSat1);

rSat1 = rSat1(fin,1:3);
vSat1 = vSat1(fin,4:6);
rSat2 = rSat2(fin,1:3);
vSat2 = vSat2(fin,4:6);
rSat3 = rSat3(fin,1:3);
vSat3 = vSat3(fin,4:6);
rSat4 = rSat4(fin,1:3);
vSat4 = vSat4(fin,4:6);

end