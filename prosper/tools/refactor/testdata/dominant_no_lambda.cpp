// Fixture for survey_sizes.py --selftest. NOT built: nothing globs tools/refactor/testdata.
//
// The discriminator for "which cursor is the lambda share measured over". `dominant` is the
// biggest region and contains ZERO lambdas, so the honest verdict is EXTRACT -- clangd can act on
// it. `sibling` is small and almost entirely one lambda body. A survey that measures the lambda
// share over the enclosing NAMESPACE instead of over the dominant region attributes sibling's
// lambda to dominant and reports HAND, which is the opposite answer: "no tool can act on it".
namespace prosper {

int dominant(int x) {
  int acc = 0;
  acc += x * 1; if (acc > 1000) acc -= 7;
  acc += x * 2; if (acc > 1000) acc -= 7;
  acc += x * 3; if (acc > 1000) acc -= 7;
  acc += x * 4; if (acc > 1000) acc -= 7;
  acc += x * 5; if (acc > 1000) acc -= 7;
  acc += x * 6; if (acc > 1000) acc -= 7;
  acc += x * 7; if (acc > 1000) acc -= 7;
  acc += x * 8; if (acc > 1000) acc -= 7;
  acc += x * 9; if (acc > 1000) acc -= 7;
  acc += x * 10; if (acc > 1000) acc -= 7;
  acc += x * 11; if (acc > 1000) acc -= 7;
  acc += x * 12; if (acc > 1000) acc -= 7;
  acc += x * 13; if (acc > 1000) acc -= 7;
  acc += x * 14; if (acc > 1000) acc -= 7;
  acc += x * 15; if (acc > 1000) acc -= 7;
  acc += x * 16; if (acc > 1000) acc -= 7;
  acc += x * 17; if (acc > 1000) acc -= 7;
  acc += x * 18; if (acc > 1000) acc -= 7;
  acc += x * 19; if (acc > 1000) acc -= 7;
  acc += x * 20; if (acc > 1000) acc -= 7;
  acc += x * 21; if (acc > 1000) acc -= 7;
  acc += x * 22; if (acc > 1000) acc -= 7;
  acc += x * 23; if (acc > 1000) acc -= 7;
  acc += x * 24; if (acc > 1000) acc -= 7;
  acc += x * 25; if (acc > 1000) acc -= 7;
  acc += x * 26; if (acc > 1000) acc -= 7;
  acc += x * 27; if (acc > 1000) acc -= 7;
  acc += x * 28; if (acc > 1000) acc -= 7;
  acc += x * 29; if (acc > 1000) acc -= 7;
  acc += x * 30; if (acc > 1000) acc -= 7;
  acc += x * 31; if (acc > 1000) acc -= 7;
  acc += x * 32; if (acc > 1000) acc -= 7;
  acc += x * 33; if (acc > 1000) acc -= 7;
  acc += x * 34; if (acc > 1000) acc -= 7;
  acc += x * 35; if (acc > 1000) acc -= 7;
  acc += x * 36; if (acc > 1000) acc -= 7;
  acc += x * 37; if (acc > 1000) acc -= 7;
  acc += x * 38; if (acc > 1000) acc -= 7;
  acc += x * 39; if (acc > 1000) acc -= 7;
  acc += x * 40; if (acc > 1000) acc -= 7;
  acc += x * 41; if (acc > 1000) acc -= 7;
  acc += x * 42; if (acc > 1000) acc -= 7;
  acc += x * 43; if (acc > 1000) acc -= 7;
  acc += x * 44; if (acc > 1000) acc -= 7;
  acc += x * 45; if (acc > 1000) acc -= 7;
  acc += x * 46; if (acc > 1000) acc -= 7;
  acc += x * 47; if (acc > 1000) acc -= 7;
  acc += x * 48; if (acc > 1000) acc -= 7;
  acc += x * 49; if (acc > 1000) acc -= 7;
  acc += x * 50; if (acc > 1000) acc -= 7;
  acc += x * 51; if (acc > 1000) acc -= 7;
  acc += x * 52; if (acc > 1000) acc -= 7;
  acc += x * 53; if (acc > 1000) acc -= 7;
  acc += x * 54; if (acc > 1000) acc -= 7;
  acc += x * 55; if (acc > 1000) acc -= 7;
  return acc;
}

int sibling(int v) {
  auto step = [](int a) {
    a += 1;
    a += 2;
    a += 3;
    a += 4;
    a += 5;
    a += 6;
    a += 7;
    a += 8;
    a += 9;
    a += 10;
    a += 11;
    a += 12;
    a += 13;
    a += 14;
    a += 15;
    a += 16;
    a += 17;
    a += 18;
    a += 19;
    a += 20;
    a += 21;
    a += 22;
    a += 23;
    a += 24;
    a += 25;
    a += 26;
    a += 27;
    a += 28;
    a += 29;
    a += 30;
    a += 31;
    a += 32;
    a += 33;
    a += 34;
    a += 35;
    a += 36;
    a += 37;
    a += 38;
    a += 39;
    a += 40;
    return a;
  };
  return step(v);
}

}  // namespace prosper
