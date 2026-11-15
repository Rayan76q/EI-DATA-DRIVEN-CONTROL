function [z_ref, psi_ref] = drone_policy_reference(P, t)
%DRONE_POLICY_REFERENCE Use the exact sampled mission exported from Python.
assert(isfield(P, 'mission_t') && isfield(P, 'mission_z_ref'), ...
    'Export the mission together with the policy.');
index = find(P.mission_t(:) <= double(t) + 1e-9, 1, 'last');
if isempty(index), index = 1; end
z_ref = P.mission_z_ref(index);
psi_ref = 0;
if isfield(P, 'mission_psi_ref'), psi_ref = P.mission_psi_ref(index); end
end
