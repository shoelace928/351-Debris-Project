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
   
R_peri = h^2/muearth*(1/(1+ecc*cos(TA)))*[cos(TA) sin(TA) 0]; % in the p hat direction 
V_peri = muearth/h * [-sin(TA) ecc+cos(TA) 0]; % in the q hat direction

QxX = [-sin(RAAN)*cos(INC)*sin(OMEGA)+cos(RAAN)*cos(OMEGA)...
-sin(RAAN)*cos(INC)*cos(OMEGA)-cos(RAAN)*sin(OMEGA) sin(RAAN)...
*sin(INC); cos(RAAN)*cos(INC)*sin(OMEGA)+sin(RAAN)*cos(OMEGA)...
cos(RAAN)*cos(INC)*cos(OMEGA)-sin(RAAN)*sin(OMEGA) -cos(RAAN)...
*sin(INC); sin(INC)*sin(OMEGA) sin(INC)*cos(OMEGA) cos(INC)];

R = transpose(QxX*transpose(R_peri));
V = transpose(QxX*transpose(V_peri));


end










