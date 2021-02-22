动态规划，英文：Dynamic Programming，简称DP，如果某一问题有很多重叠子问题，使用动态规划是最有效的。所以动态规划中每一个状态是由上一个状态推导出来的，这一点就区别于贪心，因为贪心没有状态推导，而是从局部直接选最优的

#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

 **示例 1：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 2：**

```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

 **提示：**

- `0 <= nums.length <= 100`
- `0 <= nums[i] <= 400`

**解题思路：**

1. $dp[i]$表示到了第$i$间房屋后，不管偷不偷该间房屋，口袋里拥有的最多钱数！

2. 决定$dp[i]$的因素就是第$i$间房屋偷还是不偷，而偷不偷的标准，是什么呢，当然是经历过这间房屋之后，口袋里的钱最多啦？那怎么样才能最多呢？当然是看：

   - 要偷这间房屋的话，必须保证$nums[i-1]$不能偷，则现在的钱数$dp[i]=dp[i-2]+nums[i]$
   - 不偷这间房屋的话，那么现在的金钱，即$dp[i] = dp[i-1]$
   - 所以偷还是不偷，即走过这间房屋之后，口袋里的钱有多少，即$dp[i]$为多少，当然是上述两种情况的最大值了

   状态转移方程：
   $$
   dp[i] = max(dp[i-2]+nums[i],dp[i-1])
   $$

3. $dp$数组的初始化：
   $$
   \begin{cases}
   dp[0] = nums[0] & \text{只有一间房屋则偷窃该房屋}\\
   dp[1] = max(nums[0], nums[1]) & \text{有两间房屋的话，选择金额大的进行偷窃}
   \end{cases}
   $$

4. $dp$数组的遍历：$dp[i]$是由$dp[i-2]$和$dp[i-1]$推导出来的，所以是从前向后遍历

```python
from typing import List
class Solution:
    def rob(self, nums: List[int]) -> int:
        N = len(nums)
        if N==0:    return 0
        if N==1:    return nums[0]
        dp = [0]*N      # dp[i]表示到了第i间房屋，不管偷不偷该间房屋，口袋里拥有的最多钱数
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, N):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        return dp[-1]
```

```c++
#include<iostream>
#include<vector>
using namespace std;
class Solution {
public:
    int rob(vector<int>& nums) {
        int N=nums.size();
        if(N==0)    return 0;
        if(N==1)    return nums[0];
        vector<int> dp(N);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for(int i=2; i<N; i++)
        {
            dp[i] = max(dp[i-2]+nums[i], dp[i-1]);
        }
        return dp[N-1];
    }
};
```

小结一下：

- 时间复杂度$O(N)$：需要遍历整个数组
- 空间复杂度$O(N)$：需要数组长度的$dp$数组来存取走到当前房屋时候的最大利益

参考资料：

- [@程序员Carl：动态规划：开始打家劫舍！](https://mp.weixin.qq.com/s/UZ31WdLEEFmBegdgLkJ8Dw)

- [@力扣官方题解：打家劫舍](https://leetcode-cn.com/problems/house-robber/solution/da-jia-jie-she-by-leetcode-solution/)