make -C tests/regression/spgemm clean

echo "Using N=64"
echo "SimX for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=64" ./ci/blackbox.sh --driver=simx --app=spgemm

echo "RTLSim for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=64" ./ci/blackbox.sh --driver=rtlsim --app=spgemm

make -C tests/regression/spgemm clean

echo "Using N=32"

echo "SimX for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=32" ./ci/blackbox.sh --driver=simx --app=spgemm

echo "RTLSim for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=32" ./ci/blackbox.sh --driver=rtlsim --app=spgemm
