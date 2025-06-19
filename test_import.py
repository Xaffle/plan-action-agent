try:
    from enhanced_agents import EnhancedAgent
    print('✅ 导入成功')
except Exception as e:
    print('❌ 导入失败:', e)
    import traceback
    traceback.print_exc()
