# Fragment compiler cases

CPU-only replay of one recorded fragment compiler invocation. This is deliberately separate from
frame replay: it owns compiler inputs and reproduces SOURCE words or an actual compiler refusal,
not a draw, resource upload, native subgroup decision, or rendered image. Cases may contain private
guest code and must stay local. Keep CLI verification on project-owned synthetic instructions.
