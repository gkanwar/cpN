#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor x = torch::rand({2,3}, torch::requires_grad());
  std::cout << x << "\n";
  auto y = x + 2;
  std::cout << y.grad_fn()->name() << "\n";
  auto z = 3*y*y;
  auto out = z.mean();
  std::cout << z << "\n";
  std::cout << out << "\n";

  out.backward();

  std::cout << x.grad() << "\n";
}
